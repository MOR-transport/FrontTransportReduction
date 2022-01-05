function resize(scale)

if nargin < 1
    scale = 1.5
end
s = settings;s.matlab.desktop.DisplayScaleFactor
s.matlab.desktop.DisplayScaleFactor.PersonalValue = 1.5