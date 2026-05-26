function [tOff, tOn, meanTilt, sdTilt] = present_lines(scr,visual, design, side, decision_order, mu, marker)
% present random array of lines
% - decision order is a string
% - mu is tilt from vertical in degree (positive = righward)
% 添加marker参数

% generate texture
[tex , sdTilt, meanTilt] = makeNoisyLinesTex(scr.main, visual, (-side*mu+90)/180*pi, visual.sdLines/180*pi ,visual.nLines ,visual.lineLength, visual.lineWidth, visual.textureWidth);
while sign(90 - meanTilt)~= side
    [tex , sdTilt, meanTilt] = makeNoisyLinesTex(scr.main, visual, (-side*mu+90)/180*pi, visual.sdLines/180*pi ,visual.nLines ,visual.lineLength, visual.lineWidth, visual.textureWidth);
end
meanTilt = -(meanTilt-90);

% 计算marker code：线任务，左21，右22
if side > 0
    line_code = 22;  % 右
    line_end = 28;
else
    line_code = 21;  % 左
    line_end = 27;
end

% pre-stimulus
draw_placeholders(decision_order, scr, visual);
tFix = Screen('Flip', scr.main,0);
Pres = linspace(1.2,1.5,5);%修改新增开始的刺激准备屏
design.Pres = Pres(randi(numel(Pres),1,1));
if strcmp(decision_order, '1')

    % Decision 1: 使用较长的准备时间
    WaitSecs(design.Pres);
else
    % Decision 2: 使用较短的间隔时间
    WaitSecs(design.idi);
end
% display stimulus
draw_placeholders(decision_order, scr, visual);
Screen('DrawTexture', scr.main, tex,[],visual.stimRect);
tOn = Screen('Flip', scr.main, 0);

% 打刺激开始marker
send_marker(marker, line_code);
WaitSecs(0.005);  % 保持5ms
clear_marker(marker);

% remove and take offset time for RT
draw_placeholders(decision_order, scr, visual);
tOff = Screen('Flip', scr.main, tOn + design.stimDuration);

% 打刺激结束marker
send_marker(marker, line_end);
WaitSecs(0.005);  % 保持5ms
clear_marker(marker);

% close active textures
Screen('Close', tex);
end