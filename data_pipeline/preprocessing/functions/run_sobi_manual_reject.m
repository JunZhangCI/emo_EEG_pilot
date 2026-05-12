function [EEG_cleaned, rejectedStr, IC_shown] = run_sobi_manual_reject( ...
            EEG, focus_ch, IC_shown_default)
% RUN_SOBI_MANUAL_REJECT
%   - Select EEG channels only
%   - Run SOBI ICA
%   - Interactive component inspection + rejection loop
%   - Before/after comparison on focus channels
%
% Outputs:
%   EEG_cleaned  : ICA-cleaned dataset
%   rejectedStr  : string of rejected ICs (e.g., "[1 3 5]" or "None")
%   IC_shown     : number of ICs inspected

if ~exist('IC_shown_default','var') || isempty(IC_shown_default)
    IC_shown_default = 20;
end

%% Select EEG channels only
eeg_idx = find(strcmpi({EEG.chanlocs.type}, 'EEG'));
EEG = pop_select(EEG, 'channel', eeg_idx);
EEG = eeg_checkset(EEG);
EEG_uncleaned = EEG;

%% Run SOBI ICA
EEG_wICA = pop_runica(EEG, 'icatype', 'sobi');
EEG_wICA = eeg_checkset(EEG_wICA);

%% Store to ALLEEG + push to base workspace for GUI callbacks
ALLEEG = []; CURRENTSET = 1;
[ALLEEG, EEG_wICA, CURRENTSET] = eeg_store(ALLEEG, EEG_wICA, 1);

assignin('base','EEG',EEG_wICA);
assignin('base','ALLEEG',ALLEEG);
assignin('base','CURRENTSET',CURRENTSET);

%% Component activation plot
pop_eegplot(EEG_wICA, 0, 1, 1, [], ...
            'title','ICA Component Activations');
pause;

%% Choose how many ICs to inspect
IC_shown = ask_n_IC_show(IC_shown_default);

%% Manual rejection loop
done = false;
rejectedStr = 'None';
while ~done
    EEG_cur = ALLEEG(CURRENTSET);
    %% Select components (GUI)
    assignin('base','EEG',EEG_cur);
    pop_selectcomps(EEG_cur, 1:IC_shown);
    pause;

    %% User selects ICs
    user_input = get_IC_rejection_gui();
    if ~isempty(user_input)
        comps = str2num(user_input); %#ok<ST2NM>
        EEG_cleaned = pop_subcomp(EEG_cur, comps, 0);
        EEG_cleaned = eeg_checkset(EEG_cleaned);
        rejectedStr = user_input;
    else
        EEG_cleaned = EEG_cur;
        rejectedStr = 'None';
        done = true;
        continue
    end

    %% Compare before/after on focus channels
    focus_ch_arr = strsplit(strtrim(focus_ch));
    existingCh = intersect(focus_ch_arr, {EEG_cleaned.chanlocs.labels});
    if ~isempty(existingCh)
        EEG_unc = pop_select(EEG_uncleaned, 'channel', existingCh);
        EEG_cln = pop_select(EEG_cleaned,   'channel', existingCh);
        pop_eegplot(EEG_unc,1,1,1,[], 'title','Before ICA','spacing',100);
        pop_eegplot(EEG_cln,1,1,1,[], 'title','After ICA','spacing',100);
        pause;
    else
        warning('None of the focus channels exist in this dataset.');
    end

    %% Optional grand ERP plot (epoched only)
    choice1 = questdlg( ...
        'Do you want to check the grand ERP?', ...
        'Inspect ERP', ...
        'Yes','No','No');
    if strcmp(choice1,'Yes') && ndims(EEG_cleaned.data)==3
        times = EEG_cleaned.times;   % ms
        ch_erp_unc = mean(EEG_unc.data, 3);
        grand_erp_unc = mean(ch_erp_unc, 1);
        ch_erp_cln = mean(EEG_cln.data, 3);
        grand_erp_cln = mean(ch_erp_cln, 1);
        figure;
        plot(times, grand_erp_unc, 'k', 'LineWidth',1.2); hold on;
        plot(times, grand_erp_cln, 'b', 'LineWidth',1.2);
        xline(0,'--r','LineWidth',1.5);
        hold off;
        xlabel('Time (ms)');
        ylabel('Amplitude (\muV RMS)');
        legend('Before ICA', 'After ICA');
        title(['Grand ERP, Components removed: ', rejectedStr]);
    end
    pause;
    choice2 = questdlg( ...
        'Accept cleaning?', ...
        'End ICA', ...
        'Yes','No','No');
    if strcmp(choice2,'Yes')
        done = true;
    end
end
close all;

end

%% helpers
function IC_shown = ask_n_IC_show(IC_shown_default)
valid_input = false;
while ~valid_input
    answer = inputdlg( ...
        sprintf('Enter number of components to inspect (default=%d):', IC_shown_default), ...
        'Select IC Count', ...
        [1 60]);
    % If user presses Cancel → use default and exit
    if isempty(answer)
        IC_shown = IC_shown_default;
        break
    end
    % Trim whitespace
    input_str = strtrim(answer{1});
    % If empty → use default and exit
    if isempty(input_str)
        IC_shown = IC_shown_default;
        break
    end
    % Convert to number
    IC_shown = str2double(input_str);
    % Validate
    if ~isnan(IC_shown) && IC_shown > 0
        IC_shown = round(IC_shown);   % ensure integer
        valid_input = true;
    else
        uiwait(warndlg('Please enter a valid positive number.', ...
                       'Invalid Input'));
    end
end
fprintf('Topographs of first %d components will be shown.\n', IC_shown);
end

function user_input = get_IC_rejection_gui()
% GET_IC_REJECTION_GUI
% Non-modal dialog for entering ICs to reject.
% Returns:
%   user_input : string (e.g., "[1 3 5]" or "")

user_input = '';

d = dialog('Name','Select ICs to Reject', ...
           'Position',[300 300 300 120], ...
           'WindowStyle','normal');   % NON-MODAL

uicontrol('Parent',d, ...
    'Style','text', ...
    'Position',[20 70 260 20], ...
    'String','Enter components to remove (e.g., [1 3 5]):');

editBox = uicontrol('Parent',d, ...
    'Style','edit', ...
    'Position',[20 45 260 25], ...
    'HorizontalAlignment','left');

uicontrol('Parent',d, ...
    'Position',[50 10 80 25], ...
    'String','OK', ...
    'Callback',@(src,evt) uiresume(d));

uicontrol('Parent',d, ...
    'Position',[170 10 80 25], ...
    'String','Cancel', ...
    'Callback',@(src,evt) delete(d));

uiwait(d);   % pause execution but allow figure interaction

if isvalid(editBox)
    user_input = strtrim(editBox.String);
end

if isvalid(d)
    delete(d);
end

end