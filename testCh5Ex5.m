%% 3rd Order Plant Step Response Analysis - Real and Complex Roots
% Professor-level example for instrumentation and control
% System: G(s) = K/((s+1)(s² + 4s + 13)) - Real root + Complex conjugate pair
% Author: Control Systems Analysis
% Date: 2025

clear all; close all; clc;

%% Define symbolic variables
syms s t K a2 a1 a0 real positive

% Display header
fprintf('=== 3rd Order Plant with Real and Complex Roots - Step Response Analysis ===\n\n');

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

%% Method 2: Specific Numerical Example - Real and Complex Roots
fprintf('Method 2: Specific Numerical Example - Real and Complex Roots\n');
fprintf('------------------------------------------------------------\n');

% Define 3rd order system with real and complex roots
% Real root at s = -1, Complex roots at s = -2 ± j3
% G(s) = K/((s+1)(s² + 4s + 13)) = K/(s³ + 5s² + 17s + 13)
K_val = 10;          % DC gain
a2_val = 5;          % s^2 coefficient  
a1_val = 17;         % s coefficient
a0_val = 13;         % constant term

% This gives us: G(s) = 10 / (s³ + 5s² + 17s + 13)
% Factored form: G(s) = 10 / ((s+1)(s² + 4s + 13))

fprintf('Real and complex roots example:\n');
fprintf('G(s) = %d / ((s+1)(s² + 4s + 13))\n', K_val);
fprintf('Expanded form: G(s) = %d / (s³ + %ds² + %ds + %d)\n', ...
        K_val, a2_val, a1_val, a0_val);

% Complex roots characteristics
sigma = -2;  % Real part
omega_d = 3; % Imaginary part
omega_n = sqrt(sigma^2 + omega_d^2); % Natural frequency
zeta = -sigma / omega_n;             % Damping ratio

fprintf('Root locations:\n');
fprintf('  Real root: s₁ = -1\n');
fprintf('  Complex roots: s₂,₃ = %.1f ± j%.1f\n', sigma, omega_d);
fprintf('  Natural frequency: ωₙ = %.3f rad/s\n', omega_n);
fprintf('  Damping ratio: ζ = %.3f\n', zeta);

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

%% Method 3: Partial Fraction Decomposition - Real and Complex Roots
fprintf('Method 3: Partial Fraction Analysis - Real and Complex Roots\n');
fprintf('-----------------------------------------------------------\n');

% Perform partial fraction decomposition
try
    Y_s_pfd = partfrac(Y_s_num, s);
    fprintf('Partial fraction decomposition:\n');
    fprintf('Y(s) = %s\n', char(Y_s_pfd));
catch
    fprintf('Direct partfrac failed, using manual approach...\n');
end

% Manual partial fraction for mixed roots
% Y(s) = 10/(s(s+1)(s² + 4s + 13)) = A/s + B/(s+1) + (Cs + D)/(s² + 4s + 13)
fprintf('\nManual partial fraction calculation for mixed roots:\n');
fprintf('Y(s) = 10/(s(s+1)(s² + 4s + 13))\n');
fprintf('Y(s) = A/s + B/(s+1) + (Cs + D)/(s² + 4s + 13)\n');

% Calculate residues
% A = [s*Y(s)]|_{s=0}
A = 10/((1)*(13));  % = 10/13 ≈ 0.769

% B = [(s+1)*Y(s)]|_{s=-1}  
% At s=-1: (s² + 4s + 13) = 1 - 4 + 13 = 10
B = 10/((-1)*(10));  % = -1

% For complex part, we can rewrite as: (Cs + D)/(s² + 4s + 13) = (C(s+2) + (D-2C))/((s+2)² + 9)
% This gives us exponential × [cosine + sine] terms
% Using completing the square: s² + 4s + 13 = (s+2)² + 9

% Calculate C and D by comparing coefficients or using residue method
% After algebraic manipulation:
C = -A;  % ≈ -0.769
D = -2*C - B;  % ≈ 1.538 + 1 = 2.538

fprintf('A = %.4f, B = %.4f, C = %.4f, D = %.4f\n', A, B, C, D);
fprintf('Y(s) = %.4f/s + (%.4f)/(s+1) + (%.4f*s + %.4f)/(s² + 4s + 13)\n', A, B, C, D);

% Convert to standard oscillatory form
% (Cs + D)/(s² + 4s + 13) = (C(s+2) + (D-2C))/((s+2)² + 9)
fprintf('\nStandard oscillatory form:\n');
E = C;           % Coefficient of (s+2) term
F = D - 2*C;     % Coefficient of constant term
fprintf('Y(s) = %.4f/s + (%.4f)/(s+1) + [%.4f*(s+2) + %.4f]/[(s+2)² + 9]\n', A, B, E, F);

% Time domain response
fprintf('\nTime domain components:\n');
fprintf('y(t) = %.4f*u(t) + (%.4f)*exp(-t)*u(t) + exp(-2t)*[%.4f*cos(3t) + %.4f*sin(3t)]*u(t)\n', ...
        A, B, E, F/3);

% Alternative representation using magnitude and phase
R = sqrt(E^2 + (F/3)^2);  % Magnitude
phi = atan2(F/3, E);      % Phase angle (radians)
fprintf('Alternative form: y(t) = %.4f + %.4f*exp(-t) + %.4f*exp(-2t)*cos(3t + %.3f)\n', ...
        A, B, R, phi);

%% Method 4: System Characteristics Analysis - Real and Complex Roots
fprintf('\nMethod 4: System Characteristics - Real and Complex Roots\n');
fprintf('---------------------------------------------------------\n');

% Find poles (roots of denominator)
poles_sym = solve(den_sym == 0, s);
fprintf('Symbolic poles: ');
for i = 1:length(poles_sym)
    fprintf('s%d = %s  ', i, char(poles_sym(i)));
end
fprintf('\n');

% For numerical example with real and complex roots
den_coeffs = [1, a2_val, a1_val, a0_val];  % [1, 5, 17, 13]
poles_num = roots(den_coeffs);
fprintf('Numerical poles:\n');
for i = 1:length(poles_num)
    if abs(imag(poles_num(i))) < 1e-6
        fprintf('  s%d = %.3f (real)\n', i, real(poles_num(i)));
    else
        fprintf('  s%d = %.3f %+.3fj\n', i, real(poles_num(i)), imag(poles_num(i)));
    end
end

% Classify poles
real_poles = poles_num(abs(imag(poles_num)) < 1e-6);
complex_poles = poles_num(abs(imag(poles_num)) >= 1e-6);

fprintf('\nPole classification:\n');
fprintf('Real poles: %d\n', length(real_poles));
fprintf('Complex poles: %d (in %d conjugate pairs)\n', length(complex_poles), length(complex_poles)/2);

% Complex pole characteristics
if ~isempty(complex_poles)
    % Take first complex pole (positive imaginary part)
    complex_pole = complex_poles(imag(complex_poles) > 0);
    sigma_c = real(complex_pole(1));
    omega_d_c = imag(complex_pole(1));
    omega_n_c = abs(complex_pole(1));
    zeta_c = -sigma_c / omega_n_c;
    
    fprintf('\nComplex pole characteristics:\n');
    fprintf('  σ (real part): %.3f\n', sigma_c);
    fprintf('  ωd (damped frequency): %.3f rad/s\n', omega_d_c);
    fprintf('  ωn (natural frequency): %.3f rad/s\n', omega_n_c);
    fprintf('  ζ (damping ratio): %.3f\n', zeta_c);
    
    if zeta_c < 1
        fprintf('  System type: Underdamped\n');
    elseif abs(zeta_c - 1) < 1e-6
        fprintf('  System type: Critically damped\n');
    else
        fprintf('  System type: Overdamped\n');
    end
end

% Final value theorem
final_value = limit(s * Y_s_num, s, 0);
fprintf('\nFinal value (steady-state): y(∞) = %s = %.3f\n', ...
        char(final_value), double(final_value));

%% Method 5: Time Response Plotting
fprintf('\nMethod 5: Graphical Analysis\n');
fprintf('---------------------------\n');

% Convert symbolic expression to numerical function
t_vec = 0:0.01:5;
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
title('Step Response - 3rd Order System', 'FontSize', 14);
hold on;
yline(double(final_value), 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Final Value = %.3f', double(final_value)));

% Subplot 2: Pole-zero map with complex plane
subplot(2,2,2);
hold on;

% Plot poles
real_poles_plot = poles_num(abs(imag(poles_num)) < 1e-6);
complex_poles_plot = poles_num(abs(imag(poles_num)) >= 1e-6);

% Plot real poles
if ~isempty(real_poles_plot)
    plot(real(real_poles_plot), imag(real_poles_plot), 'rx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Real Poles');
end

% Plot complex poles
if ~isempty(complex_poles_plot)
    plot(real(complex_poles_plot), imag(complex_poles_plot), 'bx', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'Complex Poles');
end

% Draw unit circle for reference (optional)
theta = 0:0.01:2*pi;
unit_circle_x = cos(theta);
unit_circle_y = sin(theta);
plot(unit_circle_x, unit_circle_y, 'k--', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);

% Draw axes
line([-5 1], [0 0], 'Color', 'k', 'LineWidth', 0.5); % Real axis
line([0 0], [-4 4], 'Color', 'k', 'LineWidth', 0.5); % Imaginary axis

grid on;
xlabel('Real Part (σ)', 'FontSize', 12);
ylabel('Imaginary Part (jω)', 'FontSize', 12);
title('Pole Locations in Complex Plane', 'FontSize', 14);
legend('Location', 'best');
axis equal;
xlim([-5, 1]);
ylim([-4, 4]);

% Add annotations for complex poles
if ~isempty(complex_poles_plot)
    for i = 1:length(complex_poles_plot)
        if imag(complex_poles_plot(i)) > 0
            text(real(complex_poles_plot(i))+0.2, imag(complex_poles_plot(i))+0.2, ...
                sprintf('%.1f+j%.1f', real(complex_poles_plot(i)), imag(complex_poles_plot(i))), ...
                'FontSize', 9);
        end
    end
end

% Subplot 3: Individual components for real and complex roots
subplot(2,2,3);
hold on;

% Use the calculated residues for mixed roots
A = 10/13;       % ≈ 0.769 (DC component)
B = -1;          % -1.000 (real exponential)
E = -10/13;      % ≈ -0.769 (cosine component)  
F = 1 + 20/13;   % ≈ 2.538 (sine component)

mode1 = A * ones(size(t_vec));                          % Step component (DC)
mode2 = B * exp(-1*t_vec);                              % Real exponential decay
mode3 = E * exp(-2*t_vec) .* cos(3*t_vec);              % Cosine × exponential
mode4 = (F/3) * exp(-2*t_vec) .* sin(3*t_vec);         % Sine × exponential
mode_complex = mode3 + mode4;                           % Combined oscillatory

plot(t_vec, mode1, 'r:', 'LineWidth', 1.5, 'DisplayName', sprintf('DC = %.3f', A));
plot(t_vec, mode2, 'g:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅e^{-t}', B));
plot(t_vec, mode3, 'b:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.3f⋅e^{-2t}cos(3t)', E));
plot(t_vec, mode4, 'm:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.3f⋅e^{-2t}sin(3t)', F/3));
plot(t_vec, mode_complex, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Combined Oscillatory');
plot(t_vec, y_values, 'k-', 'LineWidth', 2, 'DisplayName', 'Total Response');

grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Real + Complex Roots Decomposition', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);

% Subplot 4: System information for real and complex roots
subplot(2,2,4);
axis off;

% Calculate complex pole parameters for display
complex_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
if ~isempty(complex_pole)
    sigma_display = real(complex_pole(1));
    omega_d_display = imag(complex_pole(1));
    omega_n_display = abs(complex_pole(1));
    zeta_display = -sigma_display / omega_n_display;
end

info_text = {
    'System Information (Real + Complex):';
    '';
    sprintf('Transfer Function: G(s) = %d/((s+1)(s²+4s+13))', K_val);
    sprintf('Expanded: G(s) = %d/(s³+%ds²+%ds+%d)', K_val, a2_val, a1_val, a0_val);
    '';
    'Poles:';
    sprintf('  s₁ = -1.000 (real)');
    sprintf('  s₂,₃ = %.1f ± j%.1f (complex)', sigma_display, omega_d_display);
    '';
    'Complex Pole Characteristics:';
    sprintf('  ωₙ = %.3f rad/s', omega_n_display);
    sprintf('  ζ = %.3f (underdamped)', zeta_display);
    sprintf('  ωd = %.3f rad/s', omega_d_display);
    '';
    sprintf('Final Value: %.3f', double(final_value));
    '';
    'System Type: Stable (all poles in LHP)';
    'Response: Exponential + Oscillatory';
    'Behavior: Damped oscillation';
};
text(0.05, 0.95, info_text, 'FontSize', 9, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');

sgtitle('3rd Order Plant with Real and Complex Roots - Complete Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% Method 6: Performance Metrics for Oscillatory System
fprintf('Method 6: Performance Metrics - Underdamped System\n');
fprintf('--------------------------------------------------\n');

% Calculate settling time (2% criterion)
final_val = double(final_value);
tolerance = 0.02 * abs(final_val);
settling_idx = [];

% For oscillatory systems, find the last time the response exceeds the tolerance band
for i = length(y_values):-1:1
    if abs(y_values(i) - final_val) > tolerance
        settling_idx = i + 1;
        break;
    end
end

if ~isempty(settling_idx) && settling_idx <= length(t_vec)
    settling_time = t_vec(settling_idx);
    fprintf('Settling time (2%% criterion): %.3f seconds\n', settling_time);
else
    fprintf('Settling time: > %.1f seconds (beyond simulation time)\n', t_vec(end));
end

% Calculate rise time (10% to 90%)
val_10 = 0.1 * final_val;
val_90 = 0.9 * final_val;
idx_10 = find(y_values >= val_10, 1);
idx_90 = find(y_values >= val_90, 1);
if ~isempty(idx_10) && ~isempty(idx_90)
    rise_time = t_vec(idx_90) - t_vec(idx_10);
    fprintf('Rise time (10%%-90%%): %.3f seconds\n', rise_time);
end

% Calculate overshoot
max_val = max(y_values);
overshoot = ((max_val - final_val) / final_val) * 100;
fprintf('Percentage overshoot: %.2f%%\n', overshoot);

% Peak time (for oscillatory systems)
[~, peak_idx] = max(y_values);
peak_time = t_vec(peak_idx);
fprintf('Peak time: %.3f seconds\n', peak_time);

% Oscillation frequency (if applicable)
if ~isempty(complex_poles_plot)
    complex_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
    damped_freq = imag(complex_pole(1));
    oscillation_period = 2*pi / damped_freq;
    fprintf('Damped oscillation frequency: %.3f rad/s\n', damped_freq);
    fprintf('Oscillation period: %.3f seconds\n', oscillation_period);
end

fprintf('\nAnalysis complete! The system exhibits damped oscillatory behavior.\n');

%% Function for analyzing systems with real and complex roots
function analyze_mixed_root_system(K, real_root, complex_sigma, complex_omega)
    % Helper function to analyze 3rd order systems with real and complex roots
    % Example: analyze_mixed_root_system(10, -1, -2, 3) 
    % Real root at -1, complex roots at -2±j3
    
    syms s t
    
    % Construct transfer function
    % G(s) = K/((s - real_root)((s - complex_sigma)^2 + complex_omega^2))
    real_factor = (s - real_root);
    complex_factor = (s - complex_sigma)^2 + complex_omega^2;
    G = K / (real_factor * complex_factor);
    
    fprintf('Mixed roots system analysis:\n');
    fprintf('Real root: s₁ = %.1f\n', real_root);
    fprintf('Complex roots: s₂,₃ = %.1f ± j%.1f\n', complex_sigma, complex_omega);
    fprintf('G(s) = %s\n', char(G));
    
    % Step response
    Y_s = G * (1/s);
    y_t = ilaplace(Y_s, s, t);
    fprintf('Step response: y(t) = %s\n', char(simplify(y_t)));
    
    % System characteristics
    omega_n = sqrt(complex_sigma^2 + complex_omega^2);
    zeta = -complex_sigma / omega_n;
    
    fprintf('\nComplex pole characteristics:\n');
    fprintf('Natural frequency ωₙ = %.3f rad/s\n', omega_n);
    fprintf('Damping ratio ζ = %.3f\n', zeta);
    fprintf('Damped frequency ωd = %.3f rad/s\n', complex_omega);
    
    if zeta < 1
        fprintf('System type: Underdamped (oscillatory)\n');
    elseif abs(zeta - 1) < 1e-6
        fprintf('System type: Critically damped\n');
    else
        fprintf('System type: Overdamped\n');
    end
    
    fprintf('\nTime response characteristics:\n');
    fprintf('- Real exponential term: A*exp(%.1f*t)\n', real_root);
    fprintf('- Oscillatory terms: exp(%.1f*t)*[B*cos(%.1f*t) + C*sin(%.1f*t)]\n', ...
            complex_sigma, complex_omega, complex_omega);
end