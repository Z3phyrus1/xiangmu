%
% dual-decision task
%

%clear all;  clear mex;  clear functions;
addpath('functions/');
addpath('marker_need/');  % 添加marker需要的路径

home;

%% general parameters
const.gammaLinear = 0;      % use monitor linearization

%% participant informations
% collect data and, if duplicate, check before overwriting
newFile = 0;

while ~newFile
    [vpcode, dual_decision] = getVpCode;

    % create data file
    datFile = sprintf('%s.mat',vpcode);
    
    % dir names
    subDir=vpcode(1:4);
    sessionDir=vpcode(5:6);
    resdir=sprintf('data/%s/%s',subDir,sessionDir);
    
    if exist(resdir,'file')==7
        o = input('\n\n         This directory exists already. Should I continue/overwrite it [y / n]? ','s');
        if strcmp(o,'y')
            % delete files to be overwritten?
            if exist([resdir,'/',datFile])>0;                    delete([resdir,'/',datFile]); end
            if exist([resdir,'/',sprintf('%s',vpcode)])>0;       delete([resdir,'/',sprintf('%s',vpcode)]); end
            newFile = 1;
        end
    else
        mkdir(resdir);
        newFile = 1;
    end
end

%% 并口初始化
marker.enabled = true;  % 总开关false true
marker.address = hex2dec('0xCEFC');  % 请根据实际并口地址修改

% 初始化并口
if marker.enabled
    try
        config_io;
        outp(marker.address, 0);  % 初始清零
        fprintf('Marker system initialized successfully.\n');
    catch
        marker.enabled = false;
        warning('Marker system initialization failed. Running without markers.');
    end
end

%% run
sub_n = str2double(vpcode(1:2));
ses_n = str2double(vpcode(5:6));

design = genDesign(ses_n, sub_n, dual_decision, vpcode);

% prepare screens
scr = prepScreen;

% prepare stimuli
visual = prepStim(scr);

tic;
try
    % runtrials，传入marker
    design = runTrials(design, vpcode, resdir, scr, visual, marker);
catch ME
    sca;
    rethrow(ME);
end


% save updated design information
save(sprintf('./%s/%s.mat',resdir,vpcode),'design','visual','scr','const');
fprintf(1,'\n\nThis part of the experiment took %.0f min.\n',(toc)/60);

% close
sca;