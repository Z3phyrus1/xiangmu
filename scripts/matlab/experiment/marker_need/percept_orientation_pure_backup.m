function percept_orientation_pure_backup(subNum, EEG, eyetrack, nTrial , blockNum )

p.subNum = subNum;
p.nTrial = nTrial;

% make data directory
rootDir = pwd;
outputDir = fullfile(rootDir,'percept_data',filesep);
if exist(outputDir,'dir')~=7
    mkdir(outputDir)
end


if EEG
    % make a call to config_io and zero out the lpt port
    address = hex2dec('00004FF8');
    config_io
    outp(address,0);
end

% trial types
itemL_respL = 1;
itemL_respR = 2;
itemR_respL = 3;
itemR_respR = 4;


%% set up psychtoolbox stuff
% ---------------- Screen ---------------
Screen('Preference', 'SkipSyncTests', 1); %!!!
AssertOpenGL;
screens = Screen('Screens');
screenNumber = max(screens);
res = Screen('Resolution',screenNumber);
% need to change
p.resolution_width = 2560; 
p.resolution_height = 1600;
% p.refeshrate = res.hz; 
p.refeshrate = 240; %120 默认
Screen('Resolution', screenNumber, p.resolution_width, p.resolution_height, p.refeshrate);

white = WhiteIndex(screenNumber);
disp(['白色值: ', num2str(white)]);
black = BlackIndex(screenNumber);


[w, rect] = Screen('OpenWindow', screenNumber, white);
% [w, rect]=Screen('OpenWindow',screenNumber, white,[20,20,660,500]);    
[xCenter, yCenter] = RectCenter(rect);
% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', w, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
% load('.\EMM106C_EEG.mat')
% oldGammaTable = Screen('LoadNormalizedGammaTable', w , gammaTable);


% visual angle
p.viewDist = 80;
p.scnWidth = 60; % screen width, in cm
visang_rad = 2 * atan(p.scnWidth/2/p.viewDist);
visang_deg = visang_rad * (180/pi);
p.PixPerDegree = p.resolution_width / visang_deg;



% Measure the vertical refresh rate of the monitor
p.ifi = Screen('GetFlipInterval', w);
p.actual_refresh  = 1 / p.ifi;


% check that refresh rate is as expected
if abs(p.actual_refresh-p.refeshrate)>10
    error('bar_multSize_sameSeq_PTB:unexpectedRefreshRate','Refresh rate expected to be %0.03f; %0.03f measured',p.refeshrate,p.actual_refresh);
end


% Length of time and number of frames we will use for each drawing test
numSecs = 1;
numFrames = round(numSecs / p.ifi);
waitframes = 1;

HideCursor; %!!!


%----------------- keyboard ----------------------
KbName('UnifyKeyNames')
escapeKey = KbName('ESCAPE');
startKey = KbName('space');
leftKey = KbName('z');
rightKey = KbName('/?');

[keyboardIndices, ~, ~] = GetKeyboardIndices;
devID = keyboardIndices(1);


%% stimulus locations-------------------------------------------------------
p.stimulusEcc = 5.7;
stimLoc = p.stimulusEcc * p.PixPerDegree; %


%% duration 
sampleDura = 1;


%% --------------------- fixation and target ------------------
% fixation size 
p.fixSizeDeg = 0.4; 
p.fixCrossDimPix = p.fixSizeDeg * p.PixPerDegree;
xCoords = [-p.fixCrossDimPix p.fixCrossDimPix 0 0];
yCoords = [0 0 -p.fixCrossDimPix p.fixCrossDimPix];
allCoords = [xCoords; yCoords];
% Set the line width for our fixation cross
lineWidthPix = 0.12 * p.PixPerDegree;

% bar color 
p.blue = [21, 165, 234];
p.orange = [234, 74, 21];
p.green = [133, 194, 18];
p.purple = [197, 21, 234];
colors = [p.blue; p.orange; p.green; p.purple];


% stimulus size
p.lineLengDeg = 5.7;
p.lineLengPix = p.lineLengDeg * p.PixPerDegree;
p.lineWidthDeg = 0.8;
p.lineWidthPix = round(p.lineWidthDeg * p.PixPerDegree);


barRectL = CenterRectOnPointd(SetRect(0,0, p.lineWidthPix , p.lineLengPix ), xCenter - stimLoc, yCenter);
barRectR = CenterRectOnPointd(SetRect(0,0, p.lineWidthPix , p.lineLengPix ), xCenter + stimLoc, yCenter);


%% response screen
ringRect = [xCenter - (p.lineLengPix/2), yCenter-(p.lineLengPix/2), xCenter + (p.lineLengPix/2), yCenter+(p.lineLengPix/2)]; % circle for the response screen
p.circleSize = 0.3 * p.PixPerDegree;


%% set up eye tracker
if eyetrack == 1
    
    eyeOutputDir = fullfile(rootDir,'eyeData',filesep);
    if exist(eyeOutputDir,'dir')~=7
        mkdir(eyeOutputDir)
    end

    
%     % initalize the LPT port
%     address = hex2dec('00004FF8');
%     config_io
%     outp(address,0);

    dummyMode=0;       % set to 1 to initialize in dummymode (rather pointless for this example though)

    % STEP 2
    % Provide Eyelink with details about the graphics environment
    % and perform some initializations. The information is returned
    % in a structure that also contains useful defaults
    % and control codes (e.g. tracker state bit and Eyelink key values).
    el=EyelinkInitDefaults(w);

    % We are changing calibration to a gray background
    el.backgroundcolour = white; % !!!
    el.msgfontcolour  = BlackIndex(el.window);
%     el.targetbeep = 0;

    EyelinkUpdateDefaults(el);


    % STEP 3
    % Initialization of the connection with the Eyelink Gazetracker.
    % exit program if this fails.
    if ~EyelinkInit(dummyMode, 1)
        fprintf('Eyelink Init aborted.\n');
        cleanup;  % cleanup function
        return;
    end

    % make sure that we get gaze data from the Eyelink
    Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');


    % make sure we're still connected.
    if Eyelink('IsConnected')~=1 && ~dummymode
        cleanup;
        return;
    end

    % SET UP TRACKER CONFIGURATION
    % Setting the proper recording resolution, proper calibration type,
    % as well as the data file content;
    % This command is crucial to map the gaze positions from the tracker to
    % screen pixel positions to determine fixation
    Eyelink('command','screen_pixel_coords = %ld %ld %ld %ld', 0, 0, p.resolution_width-1, p.resolution_height-1);
    Eyelink('message', 'DISPLAY_COORDS %ld %ld %ld %ld', 0, 0, p.resolution_width-1, p.resolution_height-1);
    % set calibration type.
    Eyelink('command', 'calibration_type = HV9');
    Eyelink('command', 'generate_default_targets = YES');
    % set parser (conservative saccade thresholds)
    Eyelink('command', 'saccade_velocity_threshold = 35');
    Eyelink('command', 'saccade_acceleration_threshold = 9500');
    % set EDF file contents
    % 5.1 retrieve tracker version and tracker software version
    [v,vs] = Eyelink('GetTrackerVersion');
    fprintf('Running experiment on a ''%s'' tracker.\n', vs );
    vsn = regexp(vs,'\d','match');

    if v == 3 && str2double(vsn{1}) == 4 % if EL 1000 and tracker version 4.xx

        % remote mode possible add HTARGET ( head target)
        Eyelink('command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
        Eyelink('command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT,HTARGET');
        % set link data (used for gaze cursor)
        Eyelink('command', 'link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,FIXUPDATE,INPUT');
        Eyelink('command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT,HTARGET');
    else
        Eyelink('command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,FIXUPDATE,INPUT');
        Eyelink('command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS,INPUT');
        % set link data (used for gaze cursor)
        Eyelink('command', 'link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,FIXUPDATE,INPUT');
        Eyelink('command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT');
    end


    % Calibrate the eye tracker
    EyelinkDoTrackerSetup(el);

    % do a final check of calibration using driftcorrection
%     EyelinkDoDriftCorrection(el);

    % start the recording
    Eyelink('StartRecording');


end



%% experiment starts here
Screen('FillRect', w, white, rect);
[~,~,~] = DrawFormattedText(w,sprintf('Press SPACE to start'),'center','center',0);
Screen('Flip', w);
while 1
    [~,~,keyCode] = KbCheck;
    if keyCode(startKey)
        break;
    end
end


% Retreive the maximum priority number
tic
topPriorityLevel = MaxPriority(w);
Priority(topPriorityLevel);
for bb = 1 : length(blockNum)

    
    p.blockNum = blockNum(bb);
    %make a file name for this block
    fName=[outputDir, num2str(p.subNum), '_blk', num2str(p.blockNum), '.mat'];
    if exist(fName,'file')
        error('file already exists')
    end
    
    blkOnTrig = 200+p.blockNum;
    if EEG
        outp(address,blkOnTrig); % check if it is recoreded. !!!
        WaitSecs(0.006)
        outp(address,0);
    end
    

    %% stimulus orientations
    allOrients = [-70:-20 20:70];
    
    % left response and right response, left location and right location
    % need to be balanced.
    leftOrients = -70:-20; 
    rightOrients = 20:70;
    cueOri1 = [datasample(leftOrients, p.nTrial/4, true, 2) datasample(rightOrients, p.nTrial/4, true , 2)]; 
    cueOri2 = [datasample(leftOrients, p.nTrial/4, true , 2) datasample(rightOrients, p.nTrial/4, true , 2)];
    stim.leftStimOri = nan(1,p.nTrial);
    stim.rightStimOri = nan(1,p.nTrial);
    stim.leftStimOri(1:p.nTrial/2) = cueOri1; % first assign the orientations that will be cued to the left or right locations, then find orientations for the other location.
    stim.rightStimOri(p.nTrial/2+1:end) = cueOri2;
    
    for oo = 1:p.nTrial/2

%         possibleOri = allOrients - stim.leftStimOri(oo) > 10 | allOrients - stim.leftStimOri(oo) <- 10;
        if stim.leftStimOri(oo) > 0
            stim.rightStimOri(oo) = datasample(leftOrients, 1, true , 2);
        else
            stim.rightStimOri(oo) = datasample(rightOrients, 1, true , 2);
        end

    end
    
    for oo = p.nTrial/2+1:p.nTrial
        
%         possibleOri = allOrients - stim.rightStimOri(oo) > 10 | allOrients - stim.rightStimOri(oo) <- 10;
        if stim.rightStimOri(oo) > 0
            stim.leftStimOri(oo) = datasample(leftOrients, 1, true , 2);
        else
            stim.leftStimOri(oo) = datasample(rightOrients, 1, true , 2);
        end

    end
    
    randIdx = randperm(p.nTrial);
    stim.leftStimOri = stim.leftStimOri(randIdx);
    stim.rightStimOri = stim.rightStimOri(randIdx);
    
    
    %% cue 
    temp = [ones(1,p.nTrial/2), ones(1,p.nTrial/2)*2];
    stim.cue = temp(randIdx);

    
    %% stimulus color 
    tempCol = [ones(p.nTrial/2,1); ones(p.nTrial/2,1)*2];
    stim.stimColIdx = tempCol(randperm(p.nTrial)); % 1 and 2
    
    
    % record eye tracking data for each block
    if eyetrack == 1 
        edfFile = [num2str(subNum),'_b',num2str(blockNum(bb)),'.edf' ];
        fprintf('EDFFile: %s\n', edfFile );
            % open file to record data to
        res = Eyelink('Openfile', edfFile);
        if res~=0
            fprintf('Cannot create EDF file ''%s'' ', edfFile);
            cleanup;
            return;
        end
        
        EyelinkDoDriftCorrection(el); % drift correction at the beginning of each block

    end
    

    
    %% start trials
    stim.cueColor = zeros(p.nTrial,3);
    for tt = 1:p.nTrial


        stim.interTrialJitter(tt) = datasample(1:0.025:1.3, 1, true,  2);
       

        tempColor = colors(randperm(4),:);
        if stim.stimColIdx(tt) == 1
            col1 = tempColor(1,:);
            col2 = tempColor(2,:);
        elseif stim.stimColIdx(tt) == 2
            col1 = tempColor(2,:);
            col2 = tempColor(1,:);
        end
        

        stim.leftStimColor(tt,:) = col1;
        stim.rightStimColor(tt,:) = col2;  


        if stim.cue(tt) == 1  % 1 = left, 2 = right
            stim.cueColor(tt,:) = col1; 
            stim.cuedOri(tt) = stim.leftStimOri(tt);

            if stim.cuedOri(tt) < 0
                sampOnstTri = itemL_respL;
                cueOnstTri  = itemL_respL + 20;
            else
                sampOnstTri = itemL_respR;
                cueOnstTri  = itemL_respR + 20;
            end

        else
            stim.cueColor(tt,:) = col2; 
            stim.cuedOri(tt) = stim.rightStimOri(tt);

            if stim.cuedOri(tt) < 0
                sampOnstTri = itemR_respL;
                cueOnstTri  = itemR_respL + 20;
            else
                sampOnstTri = itemR_respR;
                cueOnstTri  = itemR_respR + 20;
            end

        end
        
        stim.trialType(tt) = sampOnstTri;


        % prepare stimuli for this trial
        bar1 = cat(3,ones(round(p.lineLengPix) , round(p.lineWidthPix)) .* col1(1), ones(round(p.lineLengPix) , round(p.lineWidthPix)).* col1(2), ones(round(p.lineLengPix) , round(p.lineWidthPix)).* col1(3));
        bar2 = cat(3,ones(round(p.lineLengPix) , round(p.lineWidthPix)) .* col2(1), ones(round(p.lineLengPix) , round(p.lineWidthPix)).* col2(2), ones(round(p.lineLengPix) , round(p.lineWidthPix)).* col2(3));
        
        bar1 = uint8(bar1);
        bar2 = uint8(bar2);
        
        barL = Screen('MakeTexture', w, bar1);
        barR = Screen('MakeTexture', w, bar2);
        

        stim.trialOnset(tt) = GetSecs;
        % eye tracker
        if eyetrack == 1

            % start recording eye position (preceded by a short pause so that
            % the tracker can finish the mode transition)
            % The paramerters for the 'StartRecording' call controls the
            % file_samples, file_events, link_samples, link_events availability
            %set offline for 100 ms to avoid Eye tracker freeze
            Eyelink('StopRecording');
            WaitSecs(0.1);
            Eyelink('StartRecording');
            WaitSecs(0.1);

            eye_used = Eyelink('EyeAvailable'); % to know which eye is tracked 
            if eye_used == 0
                eyeUsed = 'leftEye';
            elseif eye_used == 1
                eyeUsed = 'rightEye';
            else
                eyeUsed = 'bothEye';
            end

            Eyelink('Message', 'TrialID %d', tt);
            Eyelink('Message', 'TrialOnsetTime %d', round(stim.trialOnset(tt)*1000));
            Eyelink('Message', 'cued stim %d', stim.cue(tt)); 
            Eyelink('Message', eyeUsed);
    %         % This supplies the title at the bottom of the eyetracker display
    %         Eyelink('command', 'record_status_message "TRIAL %d/%d"', tt, 32);


        end
        

        % inter trial screen
        Screen('FillRect', w, white, rect);
        Screen('DrawLines', w, allCoords, lineWidthPix, black, [xCenter yCenter], 2);
        vbl = Screen('Flip', w);
        while GetSecs < stim.trialOnset(tt)+stim.interTrialJitter(tt)
            Screen('FillRect', w, white, rect);
            Screen('DrawLines', w, allCoords, lineWidthPix, black, [xCenter yCenter], 2);
            vbl = Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);
        end

        
        % present the samples 
        frame = 1;
        stim.stimOnset(tt) = GetSecs;
        while GetSecs < stim.stimOnset(tt)+sampleDura


            Screen('FillRect', w, white, rect);
            Screen('DrawLines', w, allCoords, lineWidthPix, black, [xCenter yCenter], 2);

            Screen('DrawTexture', w, barL,[], barRectL, stim.leftStimOri(tt) , [], [], [], []);
            Screen('DrawTexture', w, barR,[], barRectR, stim.rightStimOri(tt), [], [], [], []);
            vbl = Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);
            
            if frame == 1
                if EEG
                    outp(address,sampOnstTri); % check if it is recoreded. !!!
                end
            endKbQu
            
            [keyIsDown,secs, keyCode] = KbCheck;
            if keyCode(escapeKey)
                ShowCursor;
                Priority(0);
                sca;
                return
            end
            frame = frame + 1;
        end

        
        % cue on
        startSecs = GetSecs;
        KbQueueCreate(devID);
        KbQueueStart(devID);
        frame = 0;
        while 1 

            Screen('FillRect', w, white, rect);
            Screen('DrawLines', w, allCoords, lineWidthPix, stim.cueColor(tt,:), [xCenter yCenter], 2);
            Screen('DrawTexture', w, barL,[], barRectL, stim.leftStimOri(tt) , [], [], [], []);
            Screen('DrawTexture', w, barR,[], barRectR, stim.rightStimOri(tt), [], [], [], []);
            vbl = Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);

            if frame == 1
                if EEG 
                    outp(address,cueOnstTri); % check if it is recoreded. !!!
                end
            end

%             [pressed, firstPress, firstRelease]=KbQueueCheck(devID);
%             pressSecs = firstPress(find(firstPress)); %#ok<FNDSB>
%             if pressed
%                 if EEG 
%                     if firstPress(leftKey)
%                         outp(address, sampOnstTri + 40); % check if it is recoreded. !!!
%                     elseif firstPress(rightKey)
%                         outp(address, sampOnstTri + 44); % check if it is recoreded. !!!
%                     end
%                 end
%                 stim.keyPressed{tt} =  KbName(min(find(firstPress)));
%                 break
%             end

            [keyIsDown,pressSecs,keyCode] = KbCheck;
            if keyIsDown
                if keyCode(leftKey)
                    if EEG
                        outp(address, sampOnstTri + 40); % check if it is recoreded. !!!
                    end
                elseif keyCode(rightKey)
                    if EEG
                        outp(address, sampOnstTri + 44); % check if it is recoreded. !!!
                    end
                end
                stim.keyPressed{tt} =  KbName(keyCode);
                break;
            end
            
            frame = frame + 1;

        end
        
        tic
        tempfirstPress = keyCode;
        firstRelease = 0;
        x = 0;
        while sum(firstRelease)==0 
            
            [pressed, firstPress, firstRelease]=KbQueueCheck(devID);
%             firstPress
            % Again, fprintf will give an error if multiple keys have been pressed
%             fprintf('"%s" typed at time %.3f seconds\n', KbName(min(find(firstPress))), pressSecs - startSecs);

            if firstPress(escapeKey)
                break;
                Priority(0);
                ShowCursor('Arrow');
                Screen('CloseAll');
            end
            
            if tempfirstPress(leftKey)
                x = x - 2.8; % this takes ~825ms to rotate 90 degree
                if x/(rect(3) - rect(1))*360 <= -90
                    break
                end    
            elseif tempfirstPress(rightKey)
                x = x + 2.8;
                if x/(rect(3) - rect(1))*360 >= 90
                    break
                end
            end


            respOri = x/(rect(3) - rect(1))*360;% which orientation subjects are selecting
            [line3, line3] = make_lines_backup(p.lineLengPix, respOri, respOri, rect);
            
%             [lineCue, lineCue] = make_lines_backup(p.lineLengPix, stim.cuedOri(tt), stim.cuedOri(tt), rect);
            

    %         % show the ring to draw circles on
            Screen('FillRect', w, white, rect);
            Screen('FrameOval',w, black, ringRect)
            Screen('DrawLines', w, allCoords, lineWidthPix, stim.cueColor(tt,:), [xCenter yCenter], 2);
            Screen('glPoint', w, black, line3(1,1), line3(2,1), 10);
            Screen('glPoint', w, black, line3(1,2), line3(2,2), 10);
            Screen('glPoint', w, white, line3(1,1), line3(2,1), 8);
            Screen('glPoint', w, white, line3(1,2), line3(2,2), 8);
            
%             Screen('DrawLines', w, lineCue, p.lineWidthPix , stim.cueColor(tt,:) , stimLocL, 1, 1);

            vbl = Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);


        end
        toc

        if EEG 
            if tempfirstPress(leftKey)
                outp(address, sampOnstTri + 60); % check if it is recoreded. !!!
            elseif tempfirstPress(rightKey)
                outp(address, sampOnstTri + 64); % check if it is recoreded. !!!
            end
        end

        
        if sum(firstRelease)~=0
            releaseSecs = firstRelease(find(firstRelease)); 
            stim.dial2theEnd(tt) = 0;
        else
            releaseSecs = GetSecs;
            stim.dial2theEnd(tt) = 1; % take a note if the subject dial to the end.
        end

        
        clear tempfirstPress
        stim.cueOnset(tt) = startSecs;
        stim.respOnset(tt) = pressSecs;
        stim.respOnsetRT(tt) = pressSecs - startSecs; % response onset - cue onset, how much time does it take for the subject to start responding.
        stim.dialTime(tt) = releaseSecs - pressSecs;
        stim.respOffset(tt) = releaseSecs;
        stim.respOri(tt) = respOri;
        

%         fprintf('"%s" released at time %.3f seconds\n', KbName(min(find(firstPress))), releaseSecs - pressSecs);
        KbQueueRelease(devID);
        
        % compute the orientation difference (error)
        normDeg = mod(stim.cuedOri(tt) - respOri, 180);
        stim.errorDeg(tt) = min(180-normDeg, normDeg);
        
        if stim.errorDeg(tt)>20
            feedbackColor = [255 0 0];
        else
            feedbackColor = [0 255 0];
        end
        
        % feedback
        for frame = 1:numFrames*0.2
            Screen('FillRect', w, white, rect);
            Screen('DrawLines', w, allCoords, lineWidthPix, feedbackColor, [xCenter yCenter], 2);
            Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);
        end
        
%         % inter trial screen
%         for frame = 1:numFrames*0.5
%             Screen('FillRect', w, white, rect);
%             Screen('DrawLines', w, allCoords, lineWidthPix, black, [xCenter yCenter], 2);
% %             [~,~,~] = DrawFormattedText(w,sprintf(['Error = ',num2str(stim.errorDeg(tt))]),'center',400,0);
%             Screen('Flip', w, vbl + (waitframes - 0.5) * p.ifi);
%         end
        

    end
    
    
    blkOffTrig = 210+p.blockNum;
    if EEG
        outp(address, blkOffTrig); % check if it is recoreded. !!!
        WaitSecs(0.006)
        outp(address,0);
    end
    

    % save data file at the end of each block
    save(fName,'p','stim');
    
    if eyetrack == 1

        Eyelink('CloseFile');

        % try to download data file
        try
            fprintf('Receiving data file ''%s''\n',edfFile);
            status=Eyelink('ReceiveFile');
            if status > 0
                fprintf('ReceiveFile status %d\n',status);
            end
            movefile(edfFile, eyeOutputDir);
            if 2==exist(edfFile,'file')
                fprintf('Data file ''%s'' can be found in ''%s''\n',edfFile,pwd);
            end
        catch
            fprintf('Problem receiving data file ''%s''\n',edfFile);
        end

    end
    
    
 
%     %wait for 'space' key to continue
    Screen('FillRect', w, white, rect);
    [~,~,~] = DrawFormattedText(w,sprintf(['You have finished block ',num2str(p.blockNum)]),'center',250,0);
    [~,~,~] = DrawFormattedText(w,sprintf(['Average Error = ',num2str(mean(abs(stim.errorDeg)))]),'center',400,0);
    [~,~,~] = DrawFormattedText(w,sprintf('Press SPACE to continue'),'center',500,0);
    Screen('Flip', w);
    while 1
        [~,~,keyCode] = KbCheck;
        if keyCode(startKey)
            break;
        end
    end
    WaitSecs(.25);
    

end
toc

Priority(0);
ShowCursor('Arrow');
Screen('CloseAll');



































































































































