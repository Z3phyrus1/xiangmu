function out = mean_check_noise(sess_to_plot)
% Plot subject psychometric curves and grand-mean curves for all subjects.
% By default only sessions 03 and 04 are included.

if nargin < 1 || isempty(sess_to_plot)
    sess_to_plot = [1 2];
end

func_dir = fileparts(mfilename('fullpath'));
exp_dir = fileparts(func_dir);
data_dir = fullfile(exp_dir, 'data');

subjects = collect_all_subjects(data_dir, sess_to_plot);
if isempty(subjects)
    error('No valid subject data found in %s.', data_dir);
end

out.subjects = subjects;
out.sessions = sess_to_plot;
out.data_dir = data_dir;

out.group_mean.orientation.bias = mean(arrayfun(@(s) s.orientation.bias, subjects));
out.group_mean.orientation.sigma = mean(arrayfun(@(s) s.orientation.sigma, subjects));
out.group_mean.orientation.lapse_rate = mean(arrayfun(@(s) s.orientation.lapse_rate, subjects));
out.group_mean.motion.bias = mean(arrayfun(@(s) s.motion.bias, subjects));
out.group_mean.motion.sigma = mean(arrayfun(@(s) s.motion.sigma, subjects));
out.group_mean.motion.lapse_rate = mean(arrayfun(@(s) s.motion.lapse_rate, subjects));

[out.orientation_figure, out.orientation_curve] = plot_group_task(subjects, 'orientation', sess_to_plot);
[out.motion_figure, out.motion_curve] = plot_group_task(subjects, 'motion', sess_to_plot);
out.orientation_slope_figure = plot_slope_summary(out.orientation_curve.subject_stats, ...
    'Orientation slope');

out.motion_slope_figure = plot_slope_summary(out.motion_curve.subject_stats, ...
    'Motion slope');
end


function subjects = collect_all_subjects(data_dir, sess_to_plot)

sub_dirs = dir(fullfile(data_dir, 'sub*'));
sub_dirs = sub_dirs([sub_dirs.isdir]);
subjects = struct([]);

for iSub = 1:numel(sub_dirs)
    subject_root = fullfile(sub_dirs(iSub).folder, sub_dirs(iSub).name);
    participant_dirs = dir(fullfile(subject_root, '*'));
    participant_dirs = participant_dirs([participant_dirs.isdir]);
    participant_dirs = participant_dirs(~ismember({participant_dirs.name}, {'.', '..'}));

    for iPart = 1:numel(participant_dirs)
        vpcode = participant_dirs(iPart).name;
        participant_root = fullfile(participant_dirs(iPart).folder, vpcode);
        subject_result = collect_one_subject(participant_root, vpcode, sess_to_plot);

        if isempty(subject_result)
            continue;
        end

        if isempty(subjects)
            subjects = subject_result;
        else
            subjects(end + 1) = subject_result; %#ok<AGROW>
        end
    end
end

end


function subject_result = collect_one_subject(participant_root, vpcode, sess_to_plot)

ori_s = [];
ori_r = [];
mot_s = [];
mot_r = [];
used_sessions = [];
used_files = {};

for iSess = 1:numel(sess_to_plot)
    sess_num = sess_to_plot(iSess);
    sess_dir = fullfile(participant_root, sprintf('%02d', sess_num));

    if ~isfolder(sess_dir)
        continue;
    end

    raw_file = find_raw_trial_file(sess_dir);
    if isempty(raw_file)
        continue;
    end

    [ori_s_i, ori_r_i, mot_s_i, mot_r_i] = read_raw_trial_file(raw_file);
    ori_s = [ori_s, ori_s_i]; %#ok<AGROW>
    ori_r = [ori_r, ori_r_i]; %#ok<AGROW>
    mot_s = [mot_s, mot_s_i]; %#ok<AGROW>
    mot_r = [mot_r, mot_r_i]; %#ok<AGROW>
    used_sessions(end + 1) = sess_num; %#ok<AGROW>
    used_files{end + 1} = raw_file; %#ok<AGROW>
end

if isempty(ori_s) || isempty(mot_s)
    subject_result = [];
    return
end

subject_result = fit_subject_noise(ori_s, ori_r, mot_s, mot_r);
subject_result.vpcode = vpcode;
subject_result.sessions = used_sessions;
subject_result.files = used_files;
subject_result.orientation.stim = ori_s;
subject_result.orientation.response = ori_r;
subject_result.motion.stim = mot_s;
subject_result.motion.response = mot_r;

end


function raw_file = find_raw_trial_file(sess_dir)

files = dir(fullfile(sess_dir, '*'));
files = files(~[files.isdir]);

raw_file = '';
for iFile = 1:numel(files)
    [~, ~, ext] = fileparts(files(iFile).name);
    if isempty(ext)
        raw_file = fullfile(files(iFile).folder, files(iFile).name);
        return
    end
end

end


function [ori_s, ori_r, mot_s, mot_r] = read_raw_trial_file(filename)

ori_s = [];
ori_r = [];
mot_s = [];
mot_r = [];

ifid = fopen(filename, 'r');
if ifid == -1
    error('Cannot open file: %s', filename);
end

cleanup_obj = onCleanup(@() fclose(ifid));

while true
    line = fgetl(ifid);
    if ~ischar(line)
        break;
    end

    la = strread(line, '%s'); %#ok<STREAD>
    if numel(la) < 15
        continue;
    end

    if isnan(str2double(char(la(9))))
        signed_mu = str2double(char(la(8))) * str2double(char(la(10)));
        ori_s = [ori_s, signed_mu]; %#ok<AGROW>
        ori_r = [ori_r, str2double(char(la(15)))]; %#ok<AGROW>
    else
        signed_ch = str2double(char(la(8))) * str2double(char(la(9)));
        mot_s = [mot_s, signed_ch]; %#ok<AGROW>
        mot_r = [mot_r, str2double(char(la(15)))]; %#ok<AGROW>
    end
end

clear cleanup_obj

end


function result = fit_subject_noise(ori_s, ori_r, mot_s, mot_r)

[mu_o, sigma_o, lambda_o, ~, AIC_o] = fit_p_r(ori_s, ori_r);
[mu_o0, sigma_o0, ~, AIC_o0] = fit_p_r_0(ori_s, ori_r);
aic_w_o = calculate_akaike_weight([AIC_o, AIC_o0]);

[mu_m, sigma_m, lambda_m, ~, AIC_m] = fit_p_r(mot_s, mot_r);
[mu_m0, sigma_m0, ~, AIC_m0] = fit_p_r_0(mot_s, mot_r);
aic_w_m = calculate_akaike_weight([AIC_m, AIC_m0]);

result.orientation.bias = aic_w_o(1) * mu_o + aic_w_o(2) * mu_o0;
result.orientation.sigma = aic_w_o(1) * sigma_o + aic_w_o(2) * sigma_o0;
result.orientation.lapse_rate = aic_w_o(1) * lambda_o;
result.orientation.aic_weight = aic_w_o;

result.motion.bias = aic_w_m(1) * mu_m + aic_w_m(2) * mu_m0;
result.motion.sigma = aic_w_m(1) * sigma_m + aic_w_m(2) * sigma_m0;
result.motion.lapse_rate = aic_w_m(1) * lambda_m;
result.motion.aic_weight = aic_w_m;

end


function [fig_handle, curve_out] = plot_group_task(subjects, task_name, sess_to_plot)

[x_grid, subject_curves, subject_stats] = build_task_curves(subjects, task_name);
mean_curve = mean(subject_curves, 1, 'omitnan');

if strcmp(task_name, 'orientation')
    fig_title = sprintf('Orientation Psychometric Curves (sessions %s)', num2str(sess_to_plot));
    x_label = 'Signed orientation';
    subj_color = [0.82 0.84 0.92];
    avg_color = [0.78 0.18 0.12];
else
    fig_title = sprintf('Motion Psychometric Curves (sessions %s)', num2str(sess_to_plot));
    x_label = 'Signed coherence';
    subj_color = [0.82 0.88 0.95];
    avg_color = [0.08 0.30 0.74];
end

fig_handle = figure('Color', 'w', 'Position', [120 120 760 520]);
hold on
plot([0 0], [0 1], 'Color', [0.82 0.82 0.82], 'LineWidth', 1);
plot([x_grid(1) x_grid(end)], [0.5 0.5], 'Color', [0.82 0.82 0.82], 'LineWidth', 1);

for iSub = 1:size(subject_curves, 1)
    plot(x_grid, subject_curves(iSub, :), ...
        'Color', subj_color, ...
        'LineWidth', 0.8);

    % 在每条曲线右侧标注 slope
    text(x_grid(end), subject_curves(iSub, end), ...
        sprintf('%s: %.3f', subject_stats(iSub).vpcode, subject_stats(iSub).slope), ...
        'FontSize', 7, ...
        'Color', [0.35 0.35 0.35], ...
        'HorizontalAlignment', 'left');
end

plot(x_grid, mean_curve, 'Color', avg_color, 'LineWidth', 3);
ylim([0 1]);
xlim([x_grid(1) x_grid(end) * 1.25]);
xlabel(x_label);
ylabel('Choice probability');
title(fig_title, 'Interpreter', 'none');
box off
hold off

curve_out.x = x_grid;
curve_out.subject_curves = subject_curves;
curve_out.mean_curve = mean_curve;
curve_out.subject_stats = subject_stats;

end


function [x_grid, subject_curves, subject_stats] = build_task_curves(subjects, task_name)

nSubjects = numel(subjects);
subject_abs_max = zeros(1, nSubjects);

for iSub = 1:nSubjects
    stim = subjects(iSub).(task_name).stim;
    subject_abs_max(iSub) = max(abs(stim));
end

global_abs_max = max(subject_abs_max);
x_grid = linspace(-global_abs_max, global_abs_max, 400);
subject_curves = NaN(nSubjects, numel(x_grid));
subject_stats = struct([]);

for iSub = 1:nSubjects
    bias = subjects(iSub).(task_name).bias;
    sigma = subjects(iSub).(task_name).sigma;
    lapse_rate = subjects(iSub).(task_name).lapse_rate;

    subject_curves(iSub, :) = p_r(x_grid, bias, sigma, lapse_rate);
    slope = (1 - 2*lapse_rate) / (sigma*sqrt(2*pi));

    slope = (1 - 2*lapse_rate) / (sigma * sqrt(2*pi));
    
    subject_stats(iSub).vpcode = subjects(iSub).vpcode;
    subject_stats(iSub).bias = bias;
    subject_stats(iSub).sigma = sigma;
    subject_stats(iSub).lapse_rate = lapse_rate;
    subject_stats(iSub).slope = slope;
end

end


function p = p_r(x, mu, sigma, lambda)
% probability of choosing "+" for a cumulative Gaussian with lapse
p = lambda + (1 - 2 * lambda) * 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)));
end


function p = p_r_0(x, mu, sigma)
% probability of choosing "+" for a cumulative Gaussian
p = 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)));
end


function L = L_r(x, r, mu, sigma, lambda)
% log-likelihood of p_r
L = sum(log(p_r(x(r == 1), mu, sigma, lambda))) + ...
    sum(log(1 - p_r(x(r == 0), mu, sigma, lambda)));
end


function L = L_r_0(x, r, mu, sigma)
% log-likelihood of p_r_0
L = sum(log(p_r_0(x(r == 1), mu, sigma))) + ...
    sum(log(1 - p_r_0(x(r == 0), mu, sigma)));
end


function [mu, sigma, lambda, L, AIC] = fit_p_r(x, r, mu0, sigma0, lambda0)

if nargin < 5
    lambda0 = 0;
end
if nargin < 4
    sigma0 = mean(abs(x));
end
if nargin < 3
    mu0 = 0;
end

par0 = [mu0, sigma0, lambda0];
options = optimset('Display', 'off');
lb = [-3 * sigma0, sigma0 / 4, 0];
ub = [3 * sigma0, 4 * sigma0, 0.2];

fun = @(par) -L_r(x, r, par(1), par(2), par(3));
[par, L] = fmincon(fun, par0, [], [], [], [], lb, ub, [], options);

mu = par(1);
sigma = par(2);
lambda = par(3);
L = -L;
AIC = 2 * 3 - 2 * L;

end


function [mu, sigma, L, AIC] = fit_p_r_0(x, r, mu0, sigma0)

if nargin < 4
    sigma0 = mean(abs(x));
end
if nargin < 3
    mu0 = 0;
end

par0 = [mu0, sigma0];
options = optimset('Display', 'off');
lb = [-3 * sigma0, sigma0 / 4];
ub = [3 * sigma0, 4 * sigma0];

fun = @(par) -L_r_0(x, r, par(1), par(2));
[par, L] = fmincon(fun, par0, [], [], [], [], lb, ub, [], options);

mu = par(1);
sigma = par(2);
L = -L;
AIC = 2 * 2 - 2 * L;

end


function aic_w = calculate_akaike_weight(aic)

aic_w = aic - min(aic);
aic_w = exp(-0.5 * aic_w);
aic_w = aic_w / sum(aic_w);

end

function fig_handle = plot_slope_summary(subject_stats, fig_title)

nSub = numel(subject_stats);
vpcode = {subject_stats.vpcode};
slope = [subject_stats.slope];

fig_handle = figure('Color', 'w', 'Position', [160 160 820 420]);

subplot(1, 2, 1)
bar(slope);
set(gca, 'XTick', 1:nSub, 'XTickLabel', vpcode);
xtickangle(45);
ylabel('Maximum slope');
title([fig_title ' - bar']);
box off

subplot(1, 2, 2)
scatter(1:nSub, slope, 60, 'filled');
hold on
plot([0.5 nSub+0.5], [mean(slope) mean(slope)], 'k--', 'LineWidth', 1);
hold off
set(gca, 'XTick', 1:nSub, 'XTickLabel', vpcode);
xtickangle(45);
ylabel('Maximum slope');
title([fig_title ' - scatter']);
xlim([0.5 nSub+0.5]);
box off

end