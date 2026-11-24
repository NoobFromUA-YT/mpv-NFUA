-- Chooses how RIFE runs for each file, and changes it on the keys bound in input.conf:
--   Alt+0            automatic (what each file starts with)
--   Alt+1            off, original frame rate
--   Alt+2 .. Alt+6   a fixed multiplier (2x .. 6x)
--   Alt+8            trim mode for heights the network cannot take as they are
--   Alt+9            measure this PC again (the performance assessor, below)
-- A multiplier picked with a key lasts until the next file; the trim mode until the
-- player is closed.
--
-- Automatic: the largest whole multiplier that keeps the output at or below the screen's
-- refresh rate (screen_fps) and that the GPU can keep up with (see auto_pick).
--
-- Performance assessor: when script-opts/rife-preset.conf does not exist yet (the first
-- start, or after deleting it), the first video waits a few seconds while rife-bench
-- times the network on this GPU at 720p, 1080p, 1440p and 4K, and the screen's refresh
-- rate is read. Both go into rife-preset.conf, with the multipliers that follow from them,
-- and automatic mode uses them from then on. Alt+9 measures again.
--
-- Settings: script-opts/rife.conf (overrides the preset), or --script-opts=rife-screen_fps=120.
local options = require "mp.options"
local utils = require "mp.utils"

local o = {
    -- The screen's refresh rate (the assessor measures it).
    screen_fps = 160,
    -- 4K videos (wider than 1440p) get RIFE only up to this frame rate.
    uhd_max_fps = 30,
    -- Megapixels RIFE v4.6 interpolates per second at full and at half flow resolution.
    -- Used only when the assessor has no measurement (RTX 4070 Ti values).
    full_flow_rate = 280,
    half_flow_rate = 370,
    -- Heights the network cannot take as they are (not divisible by 32, or by 64 at half
    -- flow): "always" leaves rows out at the top and bottom to reach the multiple below,
    -- which is faster; "nearest" does that only when it is nearer than the one above,
    -- else pads; "no" always pads and shows every row.
    trim = "always",
    -- A fixed rule instead of the automatic choice (0 = off, the automatic choice applies).
    -- With e.g. 2: a video with at most rule_max_fps frames per second whose short side is at
    -- most rule_max_height gets exactly that multiplier; every other video plays as it is.
    -- YouTube / external hook example: pass 2 / 30 / 1080 (and 2x speed).
    rule_multiplier = 0,
    rule_max_fps = 30,
    rule_max_height = 1080,
    -- no: never measure by itself (Alt+9 still does).
    assess = true,
}

-- Full-resolution flow is used while it needs no more than this share of the GPU, and
-- nothing is picked automatically that needs more than MAX_LOAD of it at half flow.
local FULL_FLOW_LOAD = 0.7
local MAX_LOAD = 0.85
-- Larger than this always uses half flow, as full flow gains little there.
local FULL_FLOW_MAX_PIXELS = 2560 * 1440
-- The network takes sizes divisible by this (twice that at half flow).
local ALIGN = 32
local MIN_SIDE = 128
-- What the filter's shared engine accepts, and vf_rife's max multiplier.
local MAX_SIDE = 4096
local MAX_MULTIPLIER = 16
local UHD_MIN_LONG_SIDE = 2561
-- The filter splits work longer than this many ms into pieces, with 1 ms pauses so the
-- screen can be drawn in between; that time counts too.
local SLICE_MS, SLICE_GAP_MS = 6, 1
-- Frame rates are stored with rounding (24 fps is read as 24.000000384).
local FPS_ALLOWANCE = 0.5
-- The pauses are dropped when, with them, the work would need more than this share of
-- the GPU: 4K 30 fps at 2x takes 35 ms a frame with pauses and 31 without, and a frame
-- may take 33. Falling behind is worse than the uneven refreshes the pauses prevent.
local SLICE_MAX_LOAD = 0.8

-- The assessor's file and what it times.
local PRESET = "~~/script-opts/rife-preset.conf"
local BENCH_SIZES = "1280x720,1920x1080,2560x1440,3840x2160"
-- Shown in the preset file and after Alt+9: what automatic mode picks for these.
local TABLE_SIZES = { { "720p", 1280, 720 }, { "1080p", 1920, 1080 },
                      { "1440p", 2560, 1440 }, { "4K", 3840, 2160 } }
local TABLE_FPS = { 24, 25, 30, 50, 60 }

local video = nil   -- { w, h, fps } of the current file
local mode = nil    -- { fps = output rate } or { n = multiplier }, nil = off
-- Measured GPU ms per interpolated frame: { full = { {mp, ms}, ... }, half = ... },
-- sorted by tensor megapixels. nil: none, the rates above are used.
local timings = nil
local preset_info = nil   -- { gpu =, date =, failed = } from the preset file
local assessing = false
local pending_assess = {}
local bench_cmd = nil
local current_speed = 1
local speed_paused_auto = false
local last_vf_spec = nil

----------------------------------------------------------------------------------------
-- Preset file

local function expand(path)
    return mp.command_native({ "expand-path", path })
end

local function parse_points(s)
    local pts = {}
    for mp_, ms in (s or ""):gmatch("([%d%.]+):([%d%.]+)") do
        pts[#pts + 1] = { tonumber(mp_), tonumber(ms) }
    end
    table.sort(pts, function(a, b) return a[1] < b[1] end)
    return #pts > 0 and pts or nil
end

-- Reads rife-preset.conf into o and timings. Returns false if there is none.
local function load_preset()
    local f = io.open(expand(PRESET), "r")
    if not f then return false end
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

local have_preset = load_preset()
-- rife.conf and --script-opts come last, so what is set there wins over the preset.
options.read_options(o, "rife")

----------------------------------------------------------------------------------------
-- Planning

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

-- GPU ms for one interpolated frame: from the assessor's measurements (straight lines
-- between the measured sizes, in proportion beyond them), else from the rates.
local function frame_ms(v, flow)
    local align = flow == "half" and ALIGN * 2 or ALIGN
    local mpx = megapixels(v.w, v.h, align)
    local pts = timings and timings[flow]
    if not pts then
        return mpx / (flow == "half" and o.half_flow_rate or o.full_flow_rate) * 1000
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
    -- Full flow is only an option where it was measured (or nothing was measured).
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
    -- An unknown rate is taken as 30 fps.
    local rate = (src or 30) * spd
    local eff_v = { w = v.w, h = v.h, fps = rate }
    local top = math.min(MAX_MULTIPLIER, math.floor((o.screen_fps + FPS_ALLOWANCE) / rate))
    for n = top, 2, -1 do
        if select(2, plan(eff_v, rate * n)) <= MAX_LOAD then return n end
    end
    return nil
end

-- What a file gets by itself. With screen_fps 160 on the RTX 4070 Ti: 24 fps -> 6x (144),
-- 25 -> 6x (150), 30 -> 5x (150), 50 -> 3x (150), 60 -> 2x (120), 4K 24 -> 2x.
-- (The filter's fps= retiming is not used automatically.)
local function auto_pick(v, speed)
    if not fits(v) then return nil end
    if o.rule_multiplier > 1 then
        local src = (v.fps and v.fps > 0) and v.fps or 30   -- unknown rate: taken as 30 fps
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
    local full_spec = string.format("@rife:rife=%s:flow=%s:trim=%s%s%s", spec, flow, o.trim,
                                    slices and "" or ":slice=0", queue_opt)
    if full_spec ~= last_vf_spec then
        mp.commandv("no-osd", "vf", "add", full_spec)
        last_vf_spec = full_spec
    end
    return flow, load
end

local function describe(flow, load)
    if not mode then return "RIFE off (original fps)" end
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
    -- The window may not be open yet when the first video waits for the measurement.
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
        -- The connected output's current mode is the rate marked with *.
        for line in r.stdout:gmatch("[^\n]+") do
            local rate = tonumber(line:match("([%d%.]+)%*") or "")
            -- A virtual screen reports 0 Hz: no answer then.
            if rate and rate >= 20 then return math.floor(rate + 0.5) end
        end
    end
    -- Wayland fallbacks (wlroots / KDE Plasma) when xrandr is unavailable or Xwayland hides real Hz.
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

local function write_preset(gpu, failed, details)
    local f, err = io.open(expand(PRESET), "w")
    if not f then
        mp.msg.error("cannot write " .. PRESET .. ": " .. tostring(err))
        return
    end
    local date = os.date("%Y-%m-%d %H:%M")
    f:write("# Written by mpv-NFUA's performance assessor (scripts/rife.lua) on ", date, ".\n",
            "# Alt+9 in the player measures again; delete this file to measure at the next start.\n",
            "# A setting in rife.conf wins over the same one here.\n\n")
    f:write("measured=", date, "\n")
    if gpu then f:write("gpu=", gpu, "\n") end
    f:write("screen_fps=", tostring(o.screen_fps), "\n")
    if failed then
        f:write("failed=yes\n# ", ((details or "the measurement failed"):gsub("\n", " ")), "\n",
                "# Automatic mode uses full_flow_rate / half_flow_rate from rife.conf.\n")
    else
        f:write("\n# GPU time for one interpolated frame, as tensor megapixels:ms, at full and\n",
                "# half flow resolution (sizes ", BENCH_SIZES, ").\n")
        if timings.full then f:write("full=", points_text(timings.full), "\n") end
        if timings.half then f:write("half=", points_text(timings.half), "\n") end
    end
    f:write("\n# What automatic mode picks with this (video fps: multiplier), at most ",
            tostring(o.screen_fps), " fps:\n")
    for _, line in ipairs(multiplier_table()) do f:write("#   ", line, "\n") end
    f:close()
    have_preset = true
    preset_info = { gpu = gpu, date = date, failed = failed }
end

-- Runs rife-bench and stores the result. done(ok) is called afterwards.
local function assess(done, preserve_trim)
    if done then table.insert(pending_assess, done) end
    if assessing then return end
    assessing = true
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
    mp.msg.info("measuring the GPU with " .. tool)
    bench_cmd = mp.command_native_async({ name = "subprocess", playback_only = false,
        capture_stdout = true, capture_stderr = true,
        args = { tool, "--model", expand("~~/rife/rife_v4.6.onnx"),
                 "--model-half", expand("~~/rife/rife_v4.6_half.onnx"),
                 "--sizes", BENCH_SIZES,
                 "--full-max-pixels", tostring(FULL_FLOW_MAX_PIXELS) } },
        function(_, r)
            bench_cmd = nil
            local out = (r and r.stdout) or ""
            local gpu = out:match("gpu ([^\r\n]+)")
            local got = { full = {}, half = {} }
            for flow, size, mpx, ms in out:gmatch("result (%a+) (%S+) ([%d%.]+) ([%d%.]+)") do
                table.insert(got[flow], { tonumber(mpx), tonumber(ms) })
                mp.msg.info(string.format("%s flow %s: %.2f ms", flow, size, tonumber(ms)))
            end
            -- Re-read rife.conf so a screen_fps set there still wins, while
            -- keeping any runtime Alt+8 trim mode on Alt+9.
            local saved_trim = o.trim
            if screen then o.screen_fps = screen end
            options.read_options(o, "rife")
            if preserve_trim then o.trim = saved_trim end
            if #got.half == 0 then
                local why = out:match("error ([^\r\n]+)") or
                            ((r and r.stderr or ""):match("[^\r\n]+")) or "no result"
                mp.msg.error("GPU measurement failed: " .. why)
                timings = nil
                write_preset(gpu, true, why)
                finish(false, why)
                return
            end
            for _, flow in ipairs({ "full", "half" }) do
                table.sort(got[flow], function(a, b) return a[1] < b[1] end)
                if #got[flow] == 0 then got[flow] = nil end
            end
            timings = got
            write_preset(gpu, false)
            finish(true)
        end)
end

local function summary()
    local head = string.format("Performance: %s, screen %d Hz", preset_info and preset_info.gpu
                               or "GPU", o.screen_fps)
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
-- With no preset yet, the video waits for the measurement first (a few seconds).
mp.add_hook("on_preloaded", 50, function(hook)
    if (not have_preset and o.assess and bench_path()) or assessing then
        hook:defer()
        if not assessing then
            notify("mpv-NFUA: measuring this PC's GPU and screen (once, a few seconds)...")
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
            mp.osd_message("RIFE auto: this video plays as it is")
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
    mp.osd_message("mpv-NFUA: measuring this PC's GPU and screen...", 120)
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

local TRIM_MODES = { "always", "nearest", "no" }

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
        local _, shown = rows(video.h, flow == "half" and ALIGN * 2 or ALIGN)
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
