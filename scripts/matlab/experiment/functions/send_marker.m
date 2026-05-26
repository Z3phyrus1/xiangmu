function send_marker(marker, code)
% 发送marker，不自动清零
if marker.enabled
    outp(marker.address, code);
end
end