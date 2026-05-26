function [tOff, tOn, meanTilt, sdTilt] = present_RDM(scr,visual, design, side, decision_order, coherence, marker)
% present random array of lines
% - decision order is a string
% - coherence is (0, 1]
% 添加marker参数

% prepare stimulus settings
dots = visual.dots;
dots.coherence = coherence;
if side>0
    dots.direction = 90;
    dot_code = 12;  % 点任务向右
    dot_end = 18;
else
    dots.direction = -90;
    dot_code = 11;  % 点任务向左
    dot_end = 17;
end
meanTilt = NaN;
sdTilt = NaN;

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
if strcmp(decision_order, '1')
    %###########
    tOn = tFix + design.Pres;  % Decision 1：累加准备时间
else
    tOn = tFix + design.idi;   % Decision 2：累加间隔时间
end
movingDots(scr, dots, design.stimDuration, visual, decision_order, marker, dot_code, dot_end);

% remove and take offset time
draw_placeholders(decision_order, scr, visual);
tOff = Screen('Flip', scr.main, 0);
end