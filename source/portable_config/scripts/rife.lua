-- Chooses how RIFE runs for each file, and switches models / modes via input.conf:
--   Alt+0            automatic (what each file starts with)
--   Alt+1            off, original frame rate
--   Alt+2 .. Alt+6   a fixed multiplier (2x .. 6x)
--   Alt+8            trim mode for heights the network cannot take as they are
--   Alt+9            measure this PC again for the active model (the performance assessor)
--   Alt+m / Alt+M    roll to the next / previous installed RIFE model (saved across restarts)
-- A multiplier picked with a key lasts until the next file; the trim mode until the
-- player is closed; the chosen RIFE model persists in script-opts/rife-state.conf.
--
-- Settings: script-opts/rife.conf (overrides the preset & saved model), or --script-opts=rife-model=v4.26.
local options = require "mp.options"
local utils = require "mp.utils"

local o = {
    -- Empty = remember the last model chosen via Alt+m / Alt+M (defaults to v4.6).
    -- Set e.g. model=v4.26 in rife.conf or --script-opts=rife-model=v4.26 to force one at startup.
    model = "",
    -- The screen's refresh rate (the assessor measures it).
    screen_fps = 160,
    -- 4K videos (wider than 1440p) get RIFE only up to this frame rate.
    uhd_max_fps = 30,
    -- Fallback throughput (megapixels/sec) when a model has no benchmark preset yet.
    full_flow_rate = 280,
    half_flow_rate = 370,
    -- Heights the network cannot take as they are (not divisible by model align):
    -- "no" always pads and shows every row (default), "nearest" trims only when closer than padding,
    -- "always" leaves rows out at top/bottom (faster).
    trim = "no",
    -- Fixed rule instead of automatic choice (0 = off).
    rule_multiplier = 0,
    rule_max_fps = 30,
    rule_max_height = 1080,
    -- no: never measure by itself (Alt+9 still does).
    assess = true,
}

-- Full-resolution flow (when a separate _half model exists) is used while it needs no more
-- than this share of the GPU; single-ONNX models run full flow at all resolutions.
local FULL_FLOW_LOAD = 0.7
local MAX_LOAD = 0.85
local FULL_FLOW_MAX_PIXELS = 2560 * 1440
local MIN_SIDE = 128
local MAX_SIDE = 4096
local MAX_MULTIPLIER = 16
local UHD_MIN_LONG_SIDE = 2561
local SLICE_MS, SLICE_GAP_MS = 6, 1
local FPS_ALLOWANCE = 0.5
local SLICE_MAX_LOAD = 0.8

local STATE_FILE = "~~/script-opts/rife-state.conf"
local BENCH_SIZES = "1280x720,1920x1080,2560x1440,3840x2160"
local TABLE_SIZES = { { "720p", 1280, 720 }, { "1080p", 1920, 1080 },
                      { "1440p", 2560, 1440 }, { "4K", 3840, 2160 } }
local TABLE_FPS = { 24, 25, 30, 50, 60 }

-- Ordered catalog of supported models; optional ones appear when installed via add-rife-models.
-- Fallback rates reflect measured RTX 4070 Ti TensorRT 11.4 throughput in MP/s.
local MODEL_CATALOG = {
    { id = "v4.6",        label = "RIFE v4.6",        onnx = "rife_v4.6.onnx",        onnx_half = "rife_v4.6_half.onnx", align = 32,  align_half = 64, full_rate = 280, half_rate = 370 },
    { id = "v4.15_lite",  label = "RIFE v4.15-lite",  onnx = "rife_v4.15_lite.onnx",  align = 32,  full_rate = 170 },
    { id = "v4.22_lite",  label = "RIFE v4.22-lite",  onnx = "rife_v4.22_lite.onnx",  align = 32,  full_rate = 144 },
    { id = "v4.25_lite",  label = "RIFE v4.25-lite",  onnx = "rife_v4.25_lite.onnx",  align = 128, full_rate = 169 },
    { id = "v4.25",       label = "RIFE v4.25",       onnx = "rife_v4.25.onnx",       align = 32,  full_rate = 113 },
    { id = "v4.25_heavy", label = "RIFE v4.25-heavy", onnx = "rife_v4.25_heavy.onnx", align = 64,  full_rate = 139 },
    { id = "v4.26",       label = "RIFE v4.26",       onnx = "rife_v4.26.onnx",       align = 64,  full_rate = 124 },
}

local models = {}         -- installed model entries
local active_model = MODEL_CATALOG[1]
local video = nil         -- { w, h, fps } of the current file
local mode = nil          -- { fps = output rate } or { n = multiplier }, nil = off
local timings = nil       -- { full = { {mp, ms}, ... }, half = ... }
local preset_info = nil   -- { gpu =, date =, failed = }
local assessing = false
local pending_assess = {}
local bench_cmd = nil
local current_speed = 1
local speed_paused_auto = false
local last_vf_spec = nil
local osd_timer = nil

----------------------------------------------------------------------------------------
-- Model discovery, state persistence & presets

local function expand(path)
    return mp.command_native({ "expand-path", path })
end

local function file_exists(path)
    local info = utils.file_info(expand(path))
    return info and info.is_file
end

local function copy_entry(m)
    return {
        id = m.id,
        label = m.label,
        onnx = m.onnx,
        onnx_half = m.onnx_half,
        align = m.align,
        align_half = m.align_half,
        full_rate = m.full_rate,
        half_rate = m.half_rate,
    }
end

local function discover_models()
    local list = {}
    local seen = {}
    for _, entry in ipairs(MODEL_CATALOG) do
        if file_exists("~~/rife/" .. entry.onnx) then
            local m = copy_entry(entry)
            if m.onnx_half and not file_exists("~~/rife/" .. m.onnx_half) then
                m.onnx_half = nil
            end
            list[#list + 1] = m
            seen[m.onnx] = true
            if m.onnx_half then seen[m.onnx_half] = true end
        end
    end
    local files = utils.readdir(expand("~~/rife"), "files") or {}
    table.sort(files)
    for _, fname in ipairs(files) do
        if fname:match("^rife_.*%.onnx$") and not fname:match("_half%.onnx$") and not seen[fname] then
            local id = fname:gsub("^rife_", ""):gsub("%.onnx$", "")
            list[#list + 1] = {
                id = id,
                label = "RIFE " .. id:gsub("_", "-"),
                onnx = fname,
                align = 32,
                full_rate = 130,
            }
            seen[fname] = true
        end
    end
    if #list == 0 then
        list[1] = copy_entry(MODEL_CATALOG[1])
    end
    models = list
    return list
end

local function find_model(id)
    if not id or id == "" then return nil, nil end
    local norm = id:lower():gsub("^rife_", ""):gsub("%-", "_")
    for i, m in ipairs(models) do
        if m.id == id or m.id:lower() == norm or m.onnx == id then
            return m, i
        end
    end
    return nil, nil
end

local function load_state()
    local f = io.open(expand(STATE_FILE), "r")
    if not f then return nil end
    local saved = nil
    for line in f:lines() do
        local v = line:match("^%s*model%s*=%s*(.-)%s*$")
        if v and v ~= "" then saved = v end
    end
    f:close()
    return saved
end

local function save_state(model_id)
    local path = expand(STATE_FILE)
    local tmp = path .. ".tmp"
    local f, err = io.open(tmp, "w")
    if not f then
        f, err = io.open(path, "w")
        tmp = nil
    end
    if not f then
        mp.msg.warn("cannot save " .. STATE_FILE .. ": " .. tostring(err))
        return false
    end
    f:write("# Saved active RIFE model (updated by scripts/rife.lua on Alt+m / Alt+M).\n")
    f:write("model=", model_id, "\n")
    f:close()
    if tmp then
        local ok = os.rename(tmp, path)
        if not ok then
            os.remove(path)
            os.rename(tmp, path)
        end
    end
    return true
end

local function preset_path(id)
    if not id or id == "v4.6" then
        return "~~/script-opts/rife-preset.conf"
    end
    return string.format("~~/script-opts/rife-preset-%s.conf", id)
end

local function parse_points(s)
    local pts = {}
    for mp_, ms in (s or ""):gmatch("([%d%.]+):([%d%.]+)") do
        pts[#pts + 1] = { tonumber(mp_), tonumber(ms) }
    end
    table.sort(pts, function(a, b) return a[1] < b[1] end)
    return #pts > 0 and pts or nil
end

-- Reads a preset file into timings / preset_info and updates o.screen_fps when found.
local function load_preset(model_id)
    local path = preset_path(model_id or active_model.id)
    local f = io.open(expand(path), "r")
    if not f then
        timings = nil
        preset_info = nil
        -- Still inherit screen_fps from base preset if switching to an uncalibrated model.
        if model_id and model_id ~= "v4.6" then
            local base = io.open(expand(preset_path("v4.6")), "r")
            if base then
                for line in base:lines() do
                    local k, v = line:match("^%s*([%w_]+)%s*=%s*(.-)%s*$")
                    if k == "screen_fps" and tonumber(v) then
                        o.screen_fps = tonumber(v)
                    end
                end
                base:close()
            end
        end
        return false
    end
    local kv = {}
    for line in f:lines() do
        local k, v = line:match("^%s*([%w_]+)%s*=%s*(.-)%s*$")
        if k then kv[k] = v end
    end
    f:close()
    if tonumber(kv.screen_fps) then o.screen_fps = tonumber(kv.screen_fps) end
    local full, half = parse_points(kv.full), parse_points(kv.half)
    timings = (full or half) and { full = full, half = half } or nil
    preset_info = { gpu = kv.gpu, date = kv.measured, failed = kv.failed == "yes" }
    return true
end

local function engine_cached(m)
    local files = utils.readdir(expand("~~/rife"), "files") or {}
    local prefix = m.onnx .. "."
    for _, fname in ipairs(files) do
        if fname:sub(1, #prefix) == prefix and fname:match("%.engine$") then
            return true
        end
    end
    return false
end

-- Initialization order:
--   1. discover installed models
--   2. read rife.conf / CLI --script-opts to check explicit o.model
--   3. choose active model (explicit o.model -> saved rife-state.conf -> v4.6)
--   4. load that model's preset, then re-read rife.conf / CLI so user overrides win
discover_models()
options.read_options(o, "rife")
do
    local explicit = find_model(o.model)
    local saved = not explicit and find_model(load_state()) or nil
    active_model = explicit or saved or models[1]
end
local have_preset = load_preset(active_model.id)
options.read_options(o, "rife")

----------------------------------------------------------------------------------------
-- Planning

local function active_align(flow)
    if flow == "half" and active_model.onnx_half then
        return active_model.align_half or (active_model.align * 2)
    end
    return active_model.align
end

-- Rows the network works on, and how many rows of the picture are shown.
local function rows(h, align)
    local below = math.floor(h / align) * align
    local above = math.ceil(h / align) * align
    if below >= MIN_SIDE and (o.trim == "always" or
        (o.trim == "nearest" and h - below <= above - h)) then
        return below, below
    end
    return math.max(above, MIN_SIDE), h
end

local function megapixels(w, h, align)
    local pw = math.max(math.ceil(w / align) * align, MIN_SIDE)
    return pw * rows(h, align) / 1e6
end

local function fallback_rate(flow)
    if active_model.id == "v4.6" then
        return flow == "half" and o.half_flow_rate or o.full_flow_rate
    end
    if flow == "half" then
        return active_model.half_rate or o.half_flow_rate
    end
    return active_model.full_rate or o.full_flow_rate
end

-- GPU ms for one interpolated frame: from the assessor's measurements (straight lines
-- between the measured sizes, in proportion beyond them), else from fallback_rate.
local function frame_ms(v, flow)
    local align = active_align(flow)
    local mpx = megapixels(v.w, v.h, align)
    local pts = timings and timings[flow]
    if not pts then
        return mpx / fallback_rate(flow) * 1000
    end
    if mpx <= pts[1][1] then return pts[1][2] * mpx / pts[1][1] end
    for i = 2, #pts do
        local a, b = pts[i - 1], pts[i]
        if mpx <= b[1] then
            return a[2] + (b[2] - a[2]) * (mpx - a[1]) / (b[1] - a[1])
        end
    end
    local last = pts[#pts]
    return last[2] * mpx / last[1]
end

-- Share of the GPU for a given number of new frames per second at one flow resolution,
-- and whether the filter should split the work into slices with pauses.
local function load_for(v, per_second, flow)
    local ms = frame_ms(v, flow)
    if ms > SLICE_MS then
        local paused = ms + (math.ceil(ms / SLICE_MS) - 1) * SLICE_GAP_MS
        if per_second * paused / 1000 <= SLICE_MAX_LOAD then
            return per_second * paused / 1000, true
        end
        return per_second * ms / 1000, false
    end
    return per_second * ms / 1000, true
end

-- New frames per second needed to show the video at out_fps.
local function new_frames(src, out_fps)
    local ratio = out_fps / src
    local n = math.floor(ratio + 0.5)
    if n >= 1 and math.abs(ratio - n) <= 0.02 * ratio then
        return src * (n - 1)   -- a whole multiple: the original frames are kept
    end
    return out_fps             -- retimed: (almost) every output frame is new
end

-- Flow resolution and GPU load for showing v at out_fps.
local function plan(v, out_fps)
    local src = (v.fps and v.fps > 0) and v.fps or 30
    local per_second = new_frames(src, out_fps)
    if not active_model.onnx_half then
        local full, full_slices = load_for(v, per_second, "full")
        return "full", full, full_slices
    end
    local full_ok = not timings or timings.full
    if full_ok and v.w * v.h <= FULL_FLOW_MAX_PIXELS then
        local full, full_slices = load_for(v, per_second, "full")
        if full <= FULL_FLOW_LOAD then
            return "full", full, full_slices
        end
    end
    local half, half_slices = load_for(v, per_second, "half")
    return "half", half, half_slices
end

local function fits(v)
    return v.w <= MAX_SIDE and v.h <= MAX_SIDE
end

-- The largest whole multiplier that keeps the output at or below the screen's rate and
-- that the GPU can keep up with, or nil (play as it is).
local function best_multiplier(v, speed)
    if not fits(v) then return nil end
    local spd = math.max(1, speed or 1)
    local src = (v.fps and v.fps > 0) and v.fps or nil
    local long_side = math.max(v.w, v.h)
    if long_side >= UHD_MIN_LONG_SIDE and src and src * spd > o.uhd_max_fps + FPS_ALLOWANCE then
        return nil
    end
    local rate = (src or 30) * spd
    local eff_v = { w = v.w, h = v.h, fps = rate }
    local top = math.min(MAX_MULTIPLIER, math.floor((o.screen_fps + FPS_ALLOWANCE) / rate))
    for n = top, 2, -1 do
        if select(2, plan(eff_v, rate * n)) <= MAX_LOAD then return n end
    end
    return nil
end

local function auto_pick(v, speed)
    if not fits(v) then return nil end
    if o.rule_multiplier > 1 then
        local src = (v.fps and v.fps > 0) and v.fps or 30
        local ok = src <= o.rule_max_fps + FPS_ALLOWANCE and math.min(v.w, v.h) <= o.rule_max_height
        return ok and { n = math.min(MAX_MULTIPLIER, math.floor(o.rule_multiplier)) } or nil
    end
    local n = best_multiplier(v, speed)
    return n and { n = n } or nil
end

-- Lines like "1080p   24: 6x  25: 6x  30: 5x  50: 3x  60: 2x".
local function multiplier_table()
    local lines = {}
    for _, s in ipairs(TABLE_SIZES) do
        local parts = {}
        for _, fps in ipairs(TABLE_FPS) do
            local n = best_multiplier({ w = s[2], h = s[3], fps = fps }, 1)
            parts[#parts + 1] = string.format("%d: %s", fps, n and (n .. "x") or "off")
        end
        lines[#lines + 1] = string.format("%-6s %s", s[1], table.concat(parts, "  "))
    end
    return lines
end

----------------------------------------------------------------------------------------
-- Applying

local function osd(text, sec)
    if osd_timer then
        osd_timer:kill()
        osd_timer = nil
    end
    mp.osd_message(text, sec or 2.5)
    if sec and sec > 2.5 then
        osd_timer = mp.add_timeout(sec, function()
            osd_timer = nil
        end)
    end
end

local function apply()
    if not mode or not video then
        if last_vf_spec ~= "" then
            mp.command("no-osd vf remove @rife")
            last_vf_spec = ""
        end
        return nil
    end
    local src = ((video.fps and video.fps > 0) and video.fps or 30) * math.max(1, current_speed)
    local eff_v = { w = video.w, h = video.h, fps = src }
    local out_fps = mode.fps or (src * mode.n)
    local flow, load, slices = plan(eff_v, out_fps)
    local spec = mode.fps and string.format("fps=%g", mode.fps) or string.format("multiplier=%d", mode.n)
    local queue_opt = ""
    if not mode.fps and frame_ms(video, flow) > 0.75 * (1000 / out_fps) then
        queue_opt = ":queue=3"
    end
    local model_arg = "~~/rife/" .. active_model.onnx
    local half_arg = active_model.onnx_half and ("~~/rife/" .. active_model.onnx_half) or model_arg
    local align_arg = active_model.onnx_half and "auto" or tostring(active_model.align)
    local full_spec = string.format(
        "@rife:rife=%s:flow=%s:trim=%s:align=%s:model=%s:model-half=%s%s%s",
        spec, flow, o.trim, align_arg, model_arg, half_arg,
        slices and "" or ":slice=0", queue_opt
    )
    if full_spec ~= last_vf_spec then
        mp.commandv("no-osd", "vf", "add", full_spec)
        last_vf_spec = full_spec
    end
    return flow, load
end

local function describe(flow, load)
    local mtag = string.format("[%s]", active_model.label)
    if not mode then return "RIFE off (original fps)  " .. mtag end
    local s
    if mode.fps then
        s = string.format("RIFE auto: %g fps", mode.fps)
    else
        s = string.format(mode.auto and "RIFE auto: %dx" or "RIFE %dx", mode.n)
        if video and video.fps and video.fps > 0 then
            s = s .. string.format(": %.0f fps", video.fps * mode.n * math.max(1, current_speed))
        end
    end
    if flow == "half" then s = s .. "  (half flow)" end
    s = s .. "  " .. mtag
    if load and load > 1 then s = s .. "\nToo heavy for the GPU, frames will drop" end
    return s
end

local function read_video()
    local fallback = nil
    for _, track in ipairs(mp.get_property_native("track-list", {})) do
        if track.type == "video" and not track.albumart and not track.image
            and track["demux-w"] and track["demux-h"] then
            local info = { w = track["demux-w"], h = track["demux-h"], fps = track["demux-fps"] }
            if track.selected then return info end
            if not fallback then fallback = info end
        end
    end
    return fallback
end

local function auto_mode()
    local m = video and auto_pick(video, current_speed)
    if m then
        m.auto = true
        speed_paused_auto = false
    elseif video and current_speed > 1 and auto_pick(video, 1) then
        speed_paused_auto = true
    else
        speed_paused_auto = false
    end
    return m
end

----------------------------------------------------------------------------------------
-- Performance assessor

local function notify(text)
    mp.osd_message(text, 120)
    if package.config:sub(1, 1) == "/" then
        mp.command_native_async({ name = "subprocess", playback_only = false,
            args = { "notify-send", "-i", "video-x-generic", "-t", "6000", "mpv-NFUA", text } },
            function() end)
    end
end

local function bench_path()
    for _, p in ipairs({
        "~~/../bin/rife-bench",
        "~~/../rife-bench",
        "~~/../build/video/filter/rife/rife-bench",
        "~~/../rife-bench.exe",
    }) do
        local full = expand(p)
        if utils.file_info(full) then return full end
    end
    return nil
end

-- The screen's refresh rate: mpv's own reading once a window is open, else xrandr /
-- Wayland outputs on Linux or Win32_VideoController on Windows, else nil.
local function detect_screen_fps()
    local fps = mp.get_property_number("display-fps")
    if fps and fps > 20 then return math.floor(fps + 0.5) end
    if package.config:sub(1, 1) ~= "/" then
        local r = mp.command_native({ name = "subprocess", playback_only = false,
            capture_stdout = true,
            args = { "powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentRefreshRate -ge 20 } | Sort-Object CurrentRefreshRate -Descending | Select-Object -First 1 -ExpandProperty CurrentRefreshRate)" } })
        local rate = r and r.status == 0 and tonumber((r.stdout or ""):match("%d+"))
        if rate and rate >= 20 then return rate end
        return nil
    end
    local r = mp.command_native({ name = "subprocess", playback_only = false,
        capture_stdout = true, args = { "xrandr", "--current" } })
    if r and r.status == 0 and r.stdout then
        for line in r.stdout:gmatch("[^\n]+") do
            local rate = tonumber(line:match("([%d%.]+)%*") or "")
            if rate and rate >= 20 then return math.floor(rate + 0.5) end
        end
    end
    local wr = mp.command_native({ name = "subprocess", playback_only = false,
        capture_stdout = true, args = { "wlr-randr" } })
    if wr and wr.status == 0 and wr.stdout then
        for line in wr.stdout:gmatch("[^\n]+") do
            if line:find("current", 1, true) then
                local rate = tonumber(line:match("([%d%.]+)%s*Hz"))
                if rate and rate >= 20 then return math.floor(rate + 0.5) end
            end
        end
    end
    local kr = mp.command_native({ name = "subprocess", playback_only = false,
        capture_stdout = true, args = { "kscreen-doctor", "-o" } })
    if kr and kr.status == 0 and kr.stdout then
        for line in kr.stdout:gmatch("[^\n]+") do
            local rate = tonumber(line:match("@([%d%.]+)%*") or line:match("@([%d%.]+).*current"))
            if rate and rate >= 20 then return math.floor(rate + 0.5) end
        end
    end
    return nil
end

local function points_text(pts)
    local t = {}
    for _, p in ipairs(pts or {}) do t[#t + 1] = string.format("%.4f:%.3f", p[1], p[2]) end
    return table.concat(t, ",")
end

local function write_preset(model_obj, gpu, failed, details)
    local ppath = preset_path(model_obj.id)
    local f, err = io.open(expand(ppath), "w")
    if not f then
        mp.msg.error("cannot write " .. ppath .. ": " .. tostring(err))
        return
    end
    local date = os.date("%Y-%m-%d %H:%M")
    f:write("# Written by mpv-NFUA's performance assessor (scripts/rife.lua) on ", date, ".\n",
            "# Model: ", model_obj.label, " (", model_obj.onnx, ")\n",
            "# Alt+9 in the player measures again; delete this file to measure at the next start.\n",
            "# A setting in rife.conf wins over the same one here.\n\n")
    f:write("model=", model_obj.id, "\n")
    f:write("measured=", date, "\n")
    if gpu then f:write("gpu=", gpu, "\n") end
    f:write("screen_fps=", tostring(o.screen_fps), "\n")
    if failed then
        f:write("failed=yes\n# ", ((details or "the measurement failed"):gsub("\n", " ")), "\n",
                "# Automatic mode uses built-in rates.\n")
    else
        f:write("\n# GPU time for one interpolated frame, as tensor megapixels:ms (sizes ", BENCH_SIZES, ").\n")
        if timings.full then f:write("full=", points_text(timings.full), "\n") end
        if timings.half then f:write("half=", points_text(timings.half), "\n") end
    end
    f:write("\n# What automatic mode picks with ", model_obj.label, " (video fps: multiplier), at most ",
            tostring(o.screen_fps), " fps:\n")
    for _, line in ipairs(multiplier_table()) do f:write("#   ", line, "\n") end
    f:close()
    have_preset = true
    preset_info = { gpu = gpu, date = date, failed = failed }
end

-- Runs rife-bench on the currently active model and stores the result. done(ok) is called afterwards.
local function assess(done, preserve_trim)
    if done then table.insert(pending_assess, done) end
    if assessing then return end
    assessing = true
    local target_model = active_model
    local function finish(ok, why)
        assessing = false
        local cbs = pending_assess
        pending_assess = {}
        for _, cb in ipairs(cbs) do cb(ok, why) end
    end
    local tool = bench_path()
    local screen = detect_screen_fps()
    if not tool then
        mp.msg.warn("rife-bench not found; automatic mode keeps its built-in numbers")
        finish(false, "rife-bench was not found")
        return
    end
    mp.msg.info("measuring " .. target_model.label .. " with " .. tool)
    local args = {
        tool,
        "--model", expand("~~/rife/" .. target_model.onnx),
        "--align", tostring(target_model.align),
        "--sizes", BENCH_SIZES,
        "--runs", "10",
    }
    if target_model.onnx_half then
        table.insert(args, "--model-half")
        table.insert(args, expand("~~/rife/" .. target_model.onnx_half))
        table.insert(args, "--align-half")
        table.insert(args, tostring(target_model.align_half or (target_model.align * 2)))
        table.insert(args, "--full-max-pixels")
        table.insert(args, tostring(FULL_FLOW_MAX_PIXELS))
    end
    bench_cmd = mp.command_native_async({ name = "subprocess", playback_only = false,
        capture_stdout = true, capture_stderr = true, args = args },
        function(_, r)
            bench_cmd = nil
            local out = (r and r.stdout) or ""
            local gpu = out:match("gpu ([^\r\n]+)")
            local got = { full = {}, half = {} }
            for flow, size, mpx, ms in out:gmatch("result (%a+) (%S+) ([%d%.]+) ([%d%.]+)") do
                table.insert(got[flow], { tonumber(mpx), tonumber(ms) })
                mp.msg.info(string.format("%s %s flow %s: %.2f ms", target_model.label, flow, size, tonumber(ms)))
            end
            local saved_trim = o.trim
            if screen then o.screen_fps = screen end
            options.read_options(o, "rife")
            if preserve_trim then o.trim = saved_trim end
            if #got.full == 0 and #got.half == 0 then
                local why = out:match("error ([^\r\n]+)") or
                            ((r and r.stderr or ""):match("[^\r\n]+")) or "no result"
                mp.msg.error("GPU measurement failed: " .. why)
                timings = nil
                write_preset(target_model, gpu, true, why)
                finish(false, why)
                return
            end
            for _, flow in ipairs({ "full", "half" }) do
                table.sort(got[flow], function(a, b) return a[1] < b[1] end)
                if #got[flow] == 0 then got[flow] = nil end
            end
            timings = got
            write_preset(target_model, gpu, false)
            finish(true)
        end)
end

local function summary()
    local head = string.format("Performance (%s): %s, screen %d Hz",
        active_model.label,
        preset_info and preset_info.gpu or "GPU",
        o.screen_fps)
    return head .. "\n" .. table.concat(multiplier_table(), "\n")
end

----------------------------------------------------------------------------------------
-- Events and keys

local function start_file()
    video = read_video()
    current_speed = mp.get_property_number("speed", 1) or 1
    last_vf_spec = nil
    mode = auto_mode()
    local flow = apply()
    if video then
        mp.msg.info(string.format("%dx%d %s fps: %s%s", video.w, video.h,
            video.fps or "unknown", (describe():gsub("\n.*", "")),
            flow and (" " .. flow .. " flow") or ""))
    end
end

-- Runs before decoders start, so the video opens with the right filter straight away.
mp.add_hook("on_preloaded", 50, function(hook)
    if (not have_preset and o.assess and bench_path()) or assessing then
        hook:defer()
        if not assessing then
            notify(string.format("mpv-NFUA: calibrating %s on this GPU and screen (once, a few seconds)...",
                active_model.label))
        end
        assess(function(ok, why)
            if ok then
                mp.osd_message(summary(), 8)
            else
                mp.osd_message("mpv-NFUA: measuring failed (" .. tostring(why) ..
                               "), Alt+9 tries again", 8)
            end
            start_file()
            hook:cont()
        end)
        return
    end
    start_file()
end)

mp.observe_property("speed", "number", function(_, val)
    if not val or val <= 0 then return end
    current_speed = val
    if not video then return end
    if (mode and mode.auto) or (not mode and speed_paused_auto) then
        mode = auto_mode()
        apply()
    end
end)

mp.observe_property("vid", "native", function(_, _)
    local v = read_video()
    if not v then return end
    if video and v.w == video.w and v.h == video.h and v.fps == video.fps then return end
    video = v
    if (mode and mode.auto) or (not mode and speed_paused_auto) then
        mode = auto_mode()
    end
    apply()
end)

mp.register_event("shutdown", function()
    if bench_cmd then
        mp.abort_async_command(bench_cmd)
        bench_cmd = nil
    end
end)

mp.register_script_message("rife", function(arg)
    video = video or read_video()
    speed_paused_auto = false
    if arg == "auto" then
        mode = auto_mode()
        if not mode then
            apply()
            mp.osd_message("RIFE auto: this video plays as it is  [" .. active_model.label .. "]")
            return
        end
    else
        local n = tonumber(arg) or 2
        if n > 1 and video and not fits(video) then
            mp.osd_message("RIFE: this video is larger than the engine accepts")
            return
        end
        mode = n > 1 and { n = math.min(MAX_MULTIPLIER, math.floor(n)) } or nil
    end
    mp.osd_message(describe(apply()))
end)

-- Alt+m / Alt+M: roll through installed RIFE models and remember the choice across restarts.
mp.register_script_message("rife-model", function(arg)
    if assessing then
        mp.osd_message("mpv-NFUA: benchmark in progress, wait a moment...")
        return
    end
    discover_models()
    local _, cur_idx = find_model(active_model.id)
    cur_idx = cur_idx or 1

    local target, new_idx
    if arg == nil or arg == "" or arg == "cycle" or arg == "next" then
        if #models <= 1 then
            mp.osd_message(string.format(
                "RIFE model: %s (only model installed — run add-rife-models to add more)",
                active_model.label), 4)
            return
        end
        new_idx = (cur_idx % #models) + 1
        target = models[new_idx]
    elseif arg == "prev" then
        if #models <= 1 then
            mp.osd_message(string.format(
                "RIFE model: %s (only model installed — run add-rife-models to add more)",
                active_model.label), 4)
            return
        end
        new_idx = ((cur_idx - 2) % #models) + 1
        target = models[new_idx]
    else
        target, new_idx = find_model(arg)
        if not target then
            mp.osd_message("RIFE model not installed: " .. tostring(arg), 3)
            return
        end
    end

    active_model = target
    save_state(active_model.id)

    local saved_trim = o.trim
    have_preset = load_preset(active_model.id)
    options.read_options(o, "rife")
    o.trim = saved_trim

    video = video or read_video()
    local was_auto = (mode and mode.auto) or (not mode and speed_paused_auto)
    if was_auto then
        mode = auto_mode()
    end
    local flow, load = apply()

    local msg = string.format("RIFE model [%d/%d]: %s  (saved)", new_idx, #models, active_model.label)
    if video then
        msg = msg .. "\n" .. describe(flow, load)
    end
    local notes = {}
    if not engine_cached(active_model) then
        notes[#notes + 1] = "compiling TensorRT engine once..."
    end
    if not have_preset then
        notes[#notes + 1] = "uncalibrated (press Alt+9 to benchmark)"
    end
    if #notes > 0 then
        msg = msg .. "\n" .. table.concat(notes, "  |  ")
    end
    osd(msg, not engine_cached(active_model) and 6 or 3.5)
end)

-- Alt+9: measure again. Playback pauses and RIFE stops meanwhile, so the GPU is free.
mp.register_script_message("rife-assess", function()
    if assessing then return end
    if not bench_path() then
        mp.osd_message("mpv-NFUA: rife-bench is missing, cannot measure")
        return
    end
    local was_paused = mp.get_property_bool("pause", false)
    local was_auto = not mode or mode.auto or speed_paused_auto
    mp.set_property_bool("pause", true)
    mp.command("no-osd vf remove @rife")
    last_vf_spec = ""
    mp.osd_message(string.format("mpv-NFUA: measuring %s on this GPU and screen...", active_model.label), 120)
    assess(function(ok, why)
        video = video or read_video()
        if was_auto then mode = auto_mode() end
        local flow, load = apply()
        if ok then
            mp.osd_message(summary() .. "\n\n" .. describe(flow, load), 10)
        else
            mp.osd_message("mpv-NFUA: measuring failed (" .. tostring(why) .. ")\n" ..
                           describe(flow, load), 8)
        end
        mp.set_property_bool("pause", was_paused)
    end, true)
end)

local TRIM_MODES = { "no", "nearest", "always" }

-- "cycle" (or nothing) steps to the next mode; a mode's name picks it.
mp.register_script_message("rife-trim", function(arg)
    local new = arg
    if arg == nil or arg == "cycle" then
        new = TRIM_MODES[1]
        for i, name in ipairs(TRIM_MODES) do
            if name == o.trim then new = TRIM_MODES[i % #TRIM_MODES + 1] end
        end
    end
    local known = false
    for _, name in ipairs(TRIM_MODES) do known = known or name == new end
    if not known then
        mp.osd_message("RIFE: unknown trim mode " .. tostring(arg))
        return
    end
    o.trim = new
    video = video or read_video()
    local msg = "RIFE trim: " .. new
    if video and mode then
        local flow = apply()
        local _, shown = rows(video.h, active_align(flow))
        if shown < video.h then
            msg = msg .. string.format("\n%d of %d rows shown", shown, video.h)
        else
            msg = msg .. string.format("\nall %d rows shown", video.h)
        end
    elseif not mode then
        msg = msg .. "  (RIFE is off)"
    end
    mp.osd_message(msg)
end)
