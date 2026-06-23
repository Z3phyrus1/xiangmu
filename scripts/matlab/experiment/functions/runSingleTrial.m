function [data1, data2] = runSingleTrial(td, scr, visual, design, marker)
% function that run individual trials

% clear keyboard buffer
FlushEvents('KeyDown');

% define response keys
leftkey = KbName('LeftArrow');
rightkey = KbName('RightArrow');


%% Decision 1

% stimuli
if design.lines_first
    [tOff, tOn, meanTilt, sdTilt] = present_lines(scr,visual, design, td.side, '1', td.mu, marker);
else
    [tOff, tOn, meanTilt, sdTilt] = present_RDM(scr, visual, design, td.side, '1', td.ch, marker);
end

% response，传入marker和任务类型信息（design.lines_first）以及决策序号（1）
[rr, acc, tResp, ] = collect_response(td.side, leftkey,rightkey, tOff, marker, design.lines_first, 1);

% store data
if design.lines_first
    trial_mat = [1, design.lines_first, td.side, NaN, td.mu, meanTilt, sdTilt, tOn, tOff, rr, acc, tResp];
else
    trial_mat = [1, design.lines_first, td.side, td.ch, NaN, NaN, NaN, tOn, tOff, rr, acc, tResp];
end
rea_format = strcat(repmat('%.4f\t', 1,size(trial_mat,2)-1), '%.4f\n');
data1 = sprintf(rea_format, trial_mat);


%% determine second signal sign
if design.dual_decision
    if acc==1
        side2 = 1;
    else
        side2 = -1;
    end
else
    side2 = sign(-randn(1));
    acc2 = acc;
end

%% Decision 2

% stimuli
if ~design.lines_first
    [tOff2, tOn2, meanTilt, sdTilt] = present_lines(scr,visual, design, side2, '2', td.mu, marker);
else
    [tOff2, tOn2, meanTilt, sdTilt] = present_RDM(scr, visual, design, side2, '2', td.ch, marker);
end

% collect response，传入marker和任务类型信息（~design.lines_first）以及决策序号（2）
[rr2, acc2, tResp2, ] = collect_response(side2, leftkey,rightkey, tOff2, marker, ~design.lines_first, 2);

% store data (2)
if ~design.lines_first
    trial_mat = [2, design.lines_first, side2, NaN, td.mu, meanTilt, sdTilt, tOn2, tOff2, rr2, acc2, tResp2,];
else
    trial_mat = [2, design.lines_first, side2, td.ch, NaN, NaN, NaN, tOn2, tOff2, rr2, acc2, tResp2];
end
rea_format = strcat(repmat('%.4f\t', 1,size(trial_mat,2)-1), '%.4f\n');
data2 = sprintf(rea_format, trial_mat);

%% feedback
if design.dual_decision
    if ~acc2
        sound(visual.sound_acc0, visual.sound_fs, 8);
    end
else
    if ~acc || ~acc2
        sound(visual.sound_acc0, visual.sound_fs, 8);
    end
end