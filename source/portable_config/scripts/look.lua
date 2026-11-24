-- Alt+7 steps through ways of showing HDR video on this SDR screen, to compare by eye.
-- SDR video looks the same in all of them. The choice lasts until the player is closed;
-- the one to keep goes into mpv.conf (target-contrast / treat-srgb-as-power22).
local LOOKS = {
    { name = "1: deeper blacks, more colour (default)", contrast = "auto", power22 = "no" },
    { name = "2: like the old renderer",       contrast = "inf",  power22 = "auto" },
    { name = "3: gpu-next default (lifted shadows)", contrast = "auto", power22 = "auto" },
}
local current = 1

mp.register_script_message("look", function(arg)
    local n = tonumber(arg)
    current = n and LOOKS[n] and n or current % #LOOKS + 1
    local look = LOOKS[current]
    mp.set_property("target-contrast", look.contrast)
    mp.set_property("treat-srgb-as-power22", look.power22)
    mp.osd_message("HDR look " .. look.name)
end)
