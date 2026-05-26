function clear_marker(marker)
% 清零marker
if marker.enabled
    outp(marker.address, 0);
end
end