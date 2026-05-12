folder = 'C:\projects\emo_EEG\stimuli';

% Get all wav files
files = dir(fullfile(folder, '*.wav'));

durations = zeros(length(files),1);

for i = 1:length(files)
    info = audioinfo(fullfile(files(i).folder, files(i).name));
    durations(i) = info.Duration;   % duration in seconds
end

% Find min and max
[minDur, minIdx] = min(durations);
[maxDur, maxIdx] = max(durations);

fprintf('Min duration: %.3f sec (%s)\n', minDur, files(minIdx).name);
fprintf('Max duration: %.3f sec (%s)\n', maxDur, files(maxIdx).name);