%% 3rd Order Plant Step Response Analysis - Repeated Real Roots
% Professor-level example for instrumentation and control
% System: G(s) = K/(s+2)³ - Triple repeated real root at s = -2
% Author: Control Systems Analysis
% Date: 2025

clear all; close all; clc;

%% Define symbolic variables
syms s t K a2 a1 a0 real positive

% Display header
fprintf('=== 3rd Order Plant with Repeated Real Roots - Step Response Analysis ===\n\n');

%% Method 1: General Symbolic Transfer Function
fprintf('Method 1: General Symbolic Analysis\n');
fprintf('-----------------------------------\n');

% Define the 3rd order transfer function without zeros
% G(s) = K / (s^3 + a2*s^2 + a1*s + a0)
num_sym = K;
den_sym = s^3 + a2*s^2 + a1*s + a0;
G_sym = num_sym / den_sym;

fprintf('Transfer Function: G(s) = %s\n', char(G_sym));

% Step input in Laplace domain: 1/s
% Y(s) = G(s) * (1/s)
Y_s = G_sym * (1/s);
fprintf('Output Y(s) = G(s) * (1/s) = %s\n', char(Y_s));

% Symbolic inverse Laplace transform (general form)
fprintf('\nGeneral step response (symbolic):\n');
y_t_general = ilaplace(Y_s, s, t);
fprintf('y(t) = %s\n\n', char(y_t_general));

%% Method 2: Specific Numerical Example - Repeated Real Roots at s = -2
fprintf('Method 2: Specific Numerical Example - Repeated Real Roots at s = -2\n');
fprintf('-------------------------------------------------------------------\n');

% Define 3rd order system with triple repeated real root at -2
% G(s) = K/(s+2)³ = K/(s³ + 6s² + 12s + 8)
K_val = 10;          % DC gain
a2_val = 6;          % s^2 coefficient  
a1_val = 12;         % s coefficient
a0_val = 8;          % constant term

% This gives us: G(s) = 10 / (s³ + 6s² + 12s + 8)
% Factored form: G(s) = 10 / (s+2)³

fprintf('Triple repeated real root example:\n');
fprintf('G(s) = %d / (s+2)³\n', K_val);
fprintf('Expanded form: G(s) = %d / (s³ + %ds² + %ds + %d)\n', ...
        K_val, a2_val, a1_val, a0_val);

fprintf('Root locations:\n');
fprintf('  All roots: s₁ = s₂ = s₃ = -2 (triple repeated)\n');
fprintf('  Multiplicity: 3\n');
fprintf('  System type: Type 0 (no integrators)\n');
fprintf('  All poles in LHP → Stable system\n');

% Substitute numerical values
G_num = subs(G_sym, [K, a2, a1, a0], [K_val, a2_val, a1_val, a0_val]);
Y_s_num = G_num * (1/s);

fprintf('Y(s) = %s\n', char(Y_s_num));

% Calculate symbolic step response
y_t_num = ilaplace(Y_s_num, s, t);
fprintf('Step response: y(t) = %s\n', char(y_t_num));

% Simplify the expression
y_t_simplified = simplify(y_t_num);
fprintf('Simplified: y(t) = %s\n\n', char(y_t_simplified));

%% Method 3: Partial Fraction Decomposition - Repeated Real Roots
fprintf('Method 3: Partial Fraction Analysis - Repeated Real Roots\n');
fprintf('---------------------------------------------------------\n');

% Perform partial fraction decomposition
try
    Y_s_pfd = partfrac(Y_s_num, s);
    fprintf('Partial fraction decomposition:\n');
    fprintf('Y(s) = %s\n', char(Y_s_pfd));
catch
    fprintf('Direct partfrac failed, using manual approach...\n');
end

% Manual partial fraction for triple repeated roots
% Y(s) = 10/(s(s+2)³) = A/s + B/(s+2) + C/(s+2)² + D/(s+2)³
fprintf('\nManual partial fraction calculation for triple repeated roots:\n');
fprintf('Y(s) = 10/(s(s+2)³)\n');
fprintf('Y(s) = A/s + B/(s+2) + C/(s+2)² + D/(s+2)³\n');

% Calculate residues for repeated poles
% A = [s*Y(s)]|_{s=0}
A = 10/(2^3);  % = 10/8 = 1.25

% For triple repeated poles at s = -2:
% D = [(s+2)³*Y(s)]|_{s=-2}
D = 10/(-2);   % = -5

% C = d/ds[(s+2)³*Y(s)]|_{s=-2} = d/ds[10/s]|_{s=-2}
C = -10/((-2)^2);  % = -10/4 = -2.5

% B = (1/2!)*d²/ds²[(s+2)³*Y(s)]|_{s=-2} = (1/2)*d²/ds²[10/s]|_{s=-2}
B = (1/2) * 20/((-2)^3);  % = (1/2) * 20/(-8) = -1.25

fprintf('A = %.4f, B = %.4f, C = %.4f, D = %.4f\n', A, B, C, D);
fprintf('Y(s) = %.4f/s + (%.4f)/(s+2) + (%.4f)/(s+2)² + (%.4f)/(s+2)³\n', A, B, C, D);

% Time domain response for repeated real roots
fprintf('\nTime domain components:\n');
fprintf('y(t) = %.4f*u(t) + (%.4f)*exp(-2t)*u(t) + (%.4f)*t*exp(-2t)*u(t) + (%.4f)*t²/2*exp(-2t)*u(t)\n', ...
        A, B, C, D);

% Simplified form
fprintf('\nSimplified form:\n');
fprintf('y(t) = %.4f + exp(-2t)*[%.4f + %.4f*t + %.4f*t²]\n', A, B, C, D/2);

fprintf('\nNote: Repeated real roots produce polynomial terms multiplied by exponentials:\n');
fprintf('• DC component: %.4f (steady-state value)\n', A);
fprintf('• Exponential decay: exp(-2t)\n');
fprintf('• Linear growth term: t*exp(-2t)\n');
fprintf('• Quadratic growth term: t²*exp(-2t)\n');
fprintf('• All exponential terms decay with time constant τ = 0.5 seconds\n');

%% Method 4: System Characteristics Analysis - Repeated Real Roots
fprintf('\nMethod 4: System Characteristics - Repeated Real Roots\n');
fprintf('------------------------------------------------------\n');

% Find poles (roots of denominator)
poles_sym = solve(den_sym == 0, s);
fprintf('Symbolic poles: ');
for i = 1:length(poles_sym)
    fprintf('s%d = %s  ', i, char(poles_sym(i)));
end
fprintf('\n');

% For numerical example with repeated real roots
den_coeffs = [1, a2_val, a1_val, a0_val];  % [1, 6, 12, 8]
poles_num = roots(den_coeffs);
fprintf('Numerical poles (triple repeated at -2):\n');
for i = 1:length(poles_num)
    fprintf('  s%d = %.3f\n', i, poles_num(i));
end

% Check multiplicity
unique_poles = unique(round(poles_num, 6));
fprintf('\nPole analysis:\n');
for i = 1:length(unique_poles)
    multiplicity = sum(abs(poles_num - unique_poles(i)) < 1e-6);
    fprintf('  s = %.3f (multiplicity %d)\n', unique_poles(i), multiplicity);
end

fprintf('\nPole classification:\n');
fprintf('  All poles are real\n');
fprintf('  All poles have same location: s = -2\n');
fprintf('  Triple repeated root\n');
fprintf('  All poles in LHP → Stable system\n');

% Time constant analysis
tau = -1/unique_poles(1);  % Time constant = -1/pole_location
fprintf('\nTime domain characteristics:\n');
fprintf('  Time constant: τ = %.3f seconds\n', tau);
fprintf('  Settling time (2%% criterion): ≈ 4τ = %.1f seconds\n', 4*tau);
fprintf('  95%% settling time: ≈ 3τ = %.1f seconds\n', 3*tau);

% Final value theorem
final_value = limit(s * Y_s_num, s, 0);
fprintf('\nSteady-state analysis:\n');
fprintf('  Final value: y(∞) = %s = %.3f\n', char(final_value), double(final_value));
fprintf('  DC gain: K/a₀ = %d/%d = %.3f\n', K_val, a0_val, K_val/a0_val);
fprintf('  System type: Type 0 (no integrators)\n');
fprintf('  Steady-state error for step input: ess = 1/(1+Kp) where Kp = %.3f\n', K_val/a0_val);

% Error analysis
Kp = K_val / a0_val;  % Position error constant
ess_step = 1 / (1 + Kp);
fprintf('  Position error constant: Kp = %.3f\n', Kp);
fprintf('  Steady-state error (step): ess = %.4f\n', ess_step);

%% Method 5: Time Response Plotting
fprintf('\nMethod 5: Graphical Analysis\n');
fprintf('---------------------------\n');

% Convert symbolic expression to numerical function
t_vec = 0:0.01:4;  % Time range suitable for repeated real roots
y_func = matlabFunction(y_t_simplified);
y_values = y_func(t_vec);

% Create comprehensive plot
figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Step response
subplot(2,2,1);
plot(t_vec, y_values, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Step Response - Triple Repeated Real Root', 'FontSize', 14);
hold on;

% Add final value line
final_val = double(final_value);
yline(final_val, 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Final Value = %.3f', final_val));

% Add settling time markers
tau = 0.5;
settling_2percent = 4 * tau;  % 2% settling time
settling_5percent = 3 * tau;  % 5% settling time

xline(settling_2percent, 'g:', 'LineWidth', 1.5, ...
      'Label', sprintf('2%% Settling = %.1fs', settling_2percent));
xline(settling_5percent, 'm:', 'LineWidth', 1.5, ...
      'Label', sprintf('5%% Settling = %.1fs', settling_5percent));

legend('Total Response', 'Final Value', '2% Settling', '5% Settling', 'Location', 'southeast');

% Subplot 2: Pole-zero map for repeated real roots
subplot(2,2,2);
hold on;

% Plot repeated real poles with special annotation
pole_location = -2;
plot(pole_location, 0, 'rx', 'MarkerSize', 15, 'LineWidth', 4, 'DisplayName', 'Triple Repeated Pole');

% Add annotation for multiplicity
text(pole_location + 0.1, 0.1, '×3', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');

% Draw axes
line([-4 1], [0 0], 'Color', 'k', 'LineWidth', 0.5); % Real axis
line([0 0], [-2 2], 'Color', 'k', 'LineWidth', 0.5); % Imaginary axis

% Mark important points
plot(0, 0, 'ko', 'MarkerSize', 4, 'DisplayName', 'Origin');

grid on;
xlabel('Real Part (σ)', 'FontSize', 12);
ylabel('Imaginary Part (jω)', 'FontSize', 12);
title('Pole Locations - Repeated Real Roots', 'FontSize', 14);
legend('Location', 'best');
axis equal;
xlim([-4, 1]);
ylim([-2, 2]);

% Add pole location annotation
text(pole_location, -0.3, 's = -2', 'HorizontalAlignment', 'center', 'FontSize', 10);

% Subplot 3: Individual components for repeated real roots
subplot(2,2,3);
hold on;

% Use the calculated residues for repeated real roots
A = 1.25;        % 1.25 (DC component)
B = -1.25;       % -1.25 (exponential term)
C = -2.5;        % -2.5 (t*exponential term)
D = -5;          % -5 (t²*exponential term)

mode1 = A * ones(size(t_vec));                    % Step component (DC)
mode2 = B * exp(-2*t_vec);                        % Exponential decay
mode3 = C * t_vec .* exp(-2*t_vec);               % Linear × exponential
mode4 = (D/2) * (t_vec.^2) .* exp(-2*t_vec);      % Quadratic × exponential

plot(t_vec, mode1, 'r:', 'LineWidth', 1.5, 'DisplayName', sprintf('DC = %.2f', A));
plot(t_vec, mode2, 'g:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.2f⋅e^{-2t}', B));
plot(t_vec, mode3, 'b:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅t⋅e^{-2t}', C));
plot(t_vec, mode4, 'm:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅t²⋅e^{-2t}', D/2));
plot(t_vec, y_values, 'k-', 'LineWidth', 2, 'DisplayName', 'Total Response');

grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Repeated Real Roots Decomposition', 'FontSize', 14);
legend('Location', 'best');

% Subplot 4: System information for repeated real roots
subplot(2,2,4);
axis off;

% Calculate system parameters
tau = 0.5;  % Time constant = 1/2
Kp_val = K_val / a0_val;  % Position error constant
ess_val = 1 / (1 + Kp_val);  % Steady-state error

info_text = {
    'System Information (Repeated Real Roots):';
    '';
    sprintf('Transfer Function: G(s) = %d/(s+2)³', K_val);
    sprintf('Expanded: G(s) = %d/(s³+%ds²+%ds+%d)', K_val, a2_val, a1_val, a0_val);
    '';
    'Poles:';
    sprintf('  s₁ = s₂ = s₃ = -2.000 (triple)');
    '';
    'Time Domain Characteristics:';
    sprintf('  Time constant: τ = %.1f s', tau);
    sprintf('  Settling time: ≈ %.0f s', 4*tau);
    sprintf('  Final value: %.3f', double(final_value));
    '';
    'Error Analysis:';
    sprintf('  System type: Type 0');
    sprintf('  Kp = %.1f', Kp_val);
    sprintf('  ess (step) = %.4f', ess_val);
    '';
    'Response Components:';
    '• DC component';
    '• Exponential decay: e^(-2t)';
    '• Linear term: t⋅e^(-2t)';
    '• Quadratic term: t²⋅e^(-2t)';
    '';
    'Behavior: Overdamped, no overshoot';
};
text(0.05, 0.95, info_text, 'FontSize', 9, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');

sgtitle('3rd Order Plant with Repeated Real Roots - Complete Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% Method 6: Performance Metrics for Repeated Real Root System
fprintf('Method 6: Performance Metrics - Overdamped System\n');
fprintf('-------------------------------------------------\n');

% Calculate settling time (2% and 5% criteria)
final_val = double(final_value);
tolerance_2 = 0.02 * abs(final_val);
tolerance_5 = 0.05 * abs(final_val);

% Find settling times
settling_idx_2 = [];
settling_idx_5 = [];

% Find last time the response exceeds tolerance band
for i = length(y_values):-1:1
    if abs(y_values(i) - final_val) > tolerance_2
        settling_idx_2 = i + 1;
        break;
    end
end

for i = length(y_values):-1:1
    if abs(y_values(i) - final_val) > tolerance_5
        settling_idx_5 = i + 1;
        break;
    end
end

if ~isempty(settling_idx_2) && settling_idx_2 <= length(t_vec)
    settling_time_2 = t_vec(settling_idx_2);
    fprintf('Settling time (2%% criterion): %.3f seconds\n', settling_time_2);
else
    fprintf('Settling time (2%%): > %.1f seconds (beyond simulation)\n', t_vec(end));
end

if ~isempty(settling_idx_5) && settling_idx_5 <= length(t_vec)
    settling_time_5 = t_vec(settling_idx_5);
    fprintf('Settling time (5%% criterion): %.3f seconds\n', settling_time_5);
else
    fprintf('Settling time (5%%): > %.1f seconds (beyond simulation)\n', t_vec(end));
end

% Theoretical settling times for repeated poles
tau = 0.5;  % Time constant = 1/2
fprintf('Theoretical settling times:\n');
fprintf('  Time constant τ = %.1f seconds\n', tau);
fprintf('  2%% settling ≈ 4τ = %.1f seconds\n', 4*tau);
fprintf('  5%% settling ≈ 3τ = %.1f seconds\n', 3*tau);

% Calculate rise time (10% to 90%)
val_10 = 0.1 * final_val;
val_90 = 0.9 * final_val;
idx_10 = find(y_values >= val_10, 1);
idx_90 = find(y_values >= val_90, 1);
if ~isempty(idx_10) && ~isempty(idx_90)
    rise_time = t_vec(idx_90) - t_vec(idx_10);
    fprintf('Rise time (10%%-90%%): %.3f seconds\n', rise_time);
end

% Calculate overshoot (should be zero for overdamped system)
max_val = max(y_values);
overshoot = ((max_val - final_val) / final_val) * 100;
fprintf('Percentage overshoot: %.2f%% (expected ~0%% for overdamped)\n', overshoot);

% Time to reach 63.2% of final value (1 time constant)
val_63 = 0.632 * final_val;
idx_63 = find(y_values >= val_63, 1);
if ~isempty(idx_63)
    time_63 = t_vec(idx_63);
    fprintf('Time to 63.2%% of final value: %.3f seconds\n', time_63);
end

% Peak time analysis
[peak_val, peak_idx] = max(y_values);
peak_time = t_vec(peak_idx);
fprintf('Peak value: %.3f at t = %.3f seconds\n', peak_val, peak_time);

% Steady-state error analysis
Kp = K_val / a0_val;
ess_step = 1 / (1 + Kp);
fprintf('\nSteady-state error analysis:\n');
fprintf('Position error constant Kp = %.3f\n', Kp);
fprintf('Steady-state error for step input: ess = %.4f\n', ess_step);
fprintf('Steady-state error percentage: %.2f%%\n', ess_step * 100);

fprintf('\nSystem characteristics summary:\n');
fprintf('• Overdamped response (no overshoot)\n');
fprintf('• Monotonic approach to final value\n');
fprintf('• Multiple time constants due to repeated poles\n');
fprintf('• Slower initial response, then accelerated convergence\n');
fprintf('• Type 0 system with finite steady-state error\n');

fprintf('\nAnalysis complete!\n');

%% Function for analyzing systems with repeated real roots
function analyze_repeated_real_root_system(K, root_location, multiplicity)
    % Helper function to analyze 3rd order systems with repeated real roots
    % Example: analyze_repeated_real_root_system(10, -2, 3) 
    % Triple repeated real root at root_location
    
    syms s t
    
    if multiplicity == 3
        % Triple repeated root: G(s) = K/(s - root_location)^3
        G = K / (s - root_location)^3;
        fprintf('Triple repeated real root system:\n');
    elseif multiplicity == 2
        % Double repeated root + one simple root
        other_root = root_location - 1; % Place other root nearby
        G = K / ((s - root_location)^2 * (s - other_root));
        fprintf('Double repeated real root system:\n');
    else
        error('Multiplicity must be 2 or 3 for this function');
    end
    
    fprintf('Repeated root location: s = %.1f (multiplicity %d)\n', root_location, multiplicity);
    fprintf('G(s) = %s\n', char(G));
    
    % Step response
    Y_s = G * (1/s);
    y_t = ilaplace(Y_s, s, t);
    fprintf('Step response: y(t) = %s\n', char(simplify(y_t)));
    
    % System characteristics
    tau = -1/root_location;  % Time constant
    fprintf('\nTime domain characteristics:\n');
    fprintf('Time constant τ = %.3f seconds\n', tau);
    fprintf('Settling time (2%%) ≈ 4τ = %.1f seconds\n', 4*tau);
    
    % System classification
    if root_location < 0
        fprintf('System: Stable (pole in LHP)\n');
    else
        fprintf('System: Unstable (pole in RHP)\n');
    end
    
    if multiplicity == 3
        fprintf('Response type: Overdamped with polynomial terms\n');
        fprintf('Time response includes:\n');
        fprintf('- DC component\n');
        fprintf('- Exponential: exp(%.1f*t)\n', root_location);
        fprintf('- Linear term: t*exp(%.1f*t)\n', root_location);
        fprintf('- Quadratic term: t²*exp(%.1f*t)\n', root_location);
    elseif multiplicity == 2
        fprintf('Response type: Overdamped with linear term\n');
        fprintf('Time response includes:\n');
        fprintf('- DC component\n');
        fprintf('- Exponential terms\n');
        fprintf('- Linear term: t*exp(%.1f*t)\n', root_location);
    end
    
    % Error analysis
    fprintf('\nError analysis:\n');
    fprintf('System type: Type 0 (no integrators)\n');
    
    % Calculate final value and error constants
    final_val = limit(Y_s, s, 0);
    fprintf('Final value: %.3f\n', double(final_val));
    
    Kp = K / ((-root_location)^multiplicity);
    ess = 1 / (1 + Kp);
    fprintf('Position error constant Kp = %.3f\n', Kp);
    fprintf('Steady-state error (step): ess = %.4f\n', ess);
    
    fprintf('\nNote: Repeated real roots produce:\n');
    fprintf('- Slower initial response\n');
    fprintf('- No overshoot (overdamped)\n');
    fprintf('- Multiple time scales in settling\n');
    fprintf('- Polynomial terms × exponential decay\n');
end