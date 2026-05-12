function EEG = filter_EEG(EEG, hp, lp, do_DCremove)

% Remove DC offset and filter
if do_DCremove
    EEG = pop_rmbase(EEG, []);
    fprintf('DC offset removal is conducted. \n')
end

% Filter
fs = EEG.srate; % load sample rate of the bdf file
N_hp = 2 * ceil((3*fs/hp)/2); % calculate order of high pass cutoff
N_lp = 2 * ceil((3*fs/lp)/2); % calculate order of low pass cutoff
EEG = pop_firws(EEG, 'fcutoff', hp, 'ftype', 'highpass', 'wtype', 'hamming', 'forder', N_hp, 'minphase', 0);   % high-pass
EEG = pop_firws(EEG, 'fcutoff', lp, 'ftype', 'lowpass', 'wtype', 'hamming', 'forder', N_lp, 'minphase', 0);    % low-pass
EEG = eeg_checkset(EEG);

end