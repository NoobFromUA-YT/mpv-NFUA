-- + / - change the playback speed in whole steps of 1x (1x, 2x, 3x ...).
-- A speed between steps (e.g. 2.5x) goes to the next whole step in that direction.
-- Never below 1x; at most 16x.
local MIN, MAX = 1, 16

local function step(dir)
    local s = mp.get_property_number("speed", 1)
    local n
    if dir > 0 then n = math.floor(s + 1e-6) + 1 else n = math.ceil(s - 1e-6) - 1 end
    n = math.max(MIN, math.min(MAX, n))
    mp.set_property_number("speed", n)
    mp.osd_message(string.format("Speed: %dx", n))
end

mp.add_key_binding(nil, "speed-up", function() step(1) end, { repeatable = true })
mp.add_key_binding(nil, "speed-down", function() step(-1) end, { repeatable = true })
