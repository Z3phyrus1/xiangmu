function [rr, acc, tResp] = collect_response(side, leftkey, rightkey, tOff, marker, is_lines_task, decision_num)
% collect response
% 添加marker参数

FlushEvents('KeyDown');

% 初始化返回变量
tRelease = NaN;

while 1
    [keyisdown, secs, keycode] = KbCheck(-1);
    if keyisdown && (keycode(leftkey) || keycode(rightkey))
        tResp = secs - tOff;
        if keycode(leftkey)
            resp = -1;
            press_code = 1;  % 左键按下marker
        else
            resp = 1;
            press_code = 2;  % 右键按下marker
        end
        rr = (resp +1)/2;
        break;
    end
end

% 打按键按下marker
send_marker(marker, press_code);
WaitSecs(0.005);
clear_marker(marker);

% 等待按键释放
while KbCheck(-1); end
tRelease = GetSecs() - tOff;

if resp==side
    acc = 1;
else
    acc = 0;
end
end