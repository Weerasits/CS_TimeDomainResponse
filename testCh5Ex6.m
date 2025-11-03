%% 3rd Order Plant Step Response Analysis - Integrator and Complex Roots
% Professor-level example for instrumentation and control
% System: G(s) = K/(s(s² + 2s + 5)) - Integrator + Complex conjugate pair
% Author: Control Systems Analysis
% Date: 2025

clear all; close all; clc;

%% Define symbolic variables
syms s t K a2 a1 a0 real positive

% Display header
fprintf('=== 3rd Order Plant with Integrator and Complex Roots - Step Response Analysis ===\n\n');

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

%% Method 2: Specific Numerical Example - Real Root at Origin and Complex Roots
fprintf('Method 2: Specific Numerical Example - Real Root at s=0 and Complex Roots\n');
fprintf('-------------------------------------------------------------------------\n');

% Define 3rd order system with integrator and complex roots
% Real root at s = 0 (integrator), Complex roots at s = -1 ± j2
% G(s) = K/(s(s² + 2s + 5)) = K/(s³ + 2s² + 5s + 0)
K_val = 10;          % DC gain
a2_val = 2;          % s^2 coefficient  
a1_val = 5;          % s coefficient
a0_val = 0;          % constant term (zero because of integrator)

% This gives us: G(s) = 10 / (s³ + 2s² + 5s + 0)
% Factored form: G(s) = 10 / (s(s² + 2s + 5))

fprintf('Integrator + complex roots example:\n');
fprintf('G(s) = %d / (s(s² + 2s + 5))\n', K_val);
fprintf('Expanded form: G(s) = %d / (s³ + %ds² + %ds + %d)\n', ...
        K_val, a2_val, a1_val, a0_val);

% Complex roots characteristics
sigma = -1;  % Real part of complex poles
omega_d = 2; % Imaginary part of complex poles
omega_n = sqrt(sigma^2 + omega_d^2); % Natural frequency
zeta = -sigma / omega_n;             % Damping ratio

fprintf('Root locations:\n');
fprintf('  Real root: s₁ = 0 (integrator)\n');
fprintf('  Complex roots: s₂,₃ = %.1f ± j%.1f\n', sigma, omega_d);
fprintf('  Natural frequency: ωₙ = %.3f rad/s\n', omega_n);
fprintf('  Damping ratio: ζ = %.3f\n', zeta);
fprintf('  System type: Type 1 (one integrator)\n');

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

%% Method 3: Partial Fraction Decomposition - Integrator and Complex Roots
fprintf('Method 3: Partial Fraction Analysis - Integrator and Complex Roots\n');
fprintf('------------------------------------------------------------------\n');

% Perform partial fraction decomposition
try
    Y_s_pfd = partfrac(Y_s_num, s);
    fprintf('Partial fraction decomposition:\n');
    fprintf('Y(s) = %s\n', char(Y_s_pfd));
catch
    fprintf('Direct partfrac failed, using manual approach...\n');
end

% Manual partial fraction for integrator + complex roots
% Y(s) = 10/(s²(s² + 2s + 5)) = A/s² + B/s + (Cs + D)/(s² + 2s + 5)
fprintf('\nManual partial fraction calculation for integrator + complex roots:\n');
fprintf('Y(s) = 10/(s²(s² + 2s + 5))\n');
fprintf('Y(s) = A/s² + B/s + (Cs + D)/(s² + 2s + 5)\n');

% Calculate residues
% A = [s²*Y(s)]|_{s=0} = 10/(s² + 2s + 5)|_{s=0}
A = 10/5;  % = 2

% B = d/ds[s²*Y(s)]|_{s=0} = d/ds[10/(s² + 2s + 5)]|_{s=0}
% B = -10*(2s + 2)/(s² + 2s + 5)²|_{s=0} = -10*2/25 = -20/25
B = -10*2/25;  % = -0.8

% For complex part, solve by comparing coefficients or residue method
% Using s² + 2s + 5 = (s+1)² + 4, we can find C and D
% After algebraic manipulation:
C = -A;  % = -2
D = -B - 2*C;  % = 0.8 + 4 = 4.8

fprintf('A = %.4f, B = %.4f, C = %.4f, D = %.4f\n', A, B, C, D);
fprintf('Y(s) = %.4f/s² + (%.4f)/s + (%.4f*s + %.4f)/(s² + 2s + 5)\n', A, B, C, D);

% Convert to standard oscillatory form
% (Cs + D)/(s² + 2s + 5) = (C(s+1) + (D-C))/((s+1)² + 4)
fprintf('\nStandard oscillatory form:\n');
E = C;           % Coefficient of (s+1) term
F = D - C;       % Coefficient of constant term
fprintf('Y(s) = %.4f/s² + (%.4f)/s + [%.4f*(s+1) + %.4f]/[(s+1)² + 4]\n', A, B, E, F);

% Time domain response for integrator system
fprintf('\nTime domain components:\n');
fprintf('y(t) = %.4f*t*u(t) + (%.4f)*u(t) + exp(-t)*[%.4f*cos(2t) + %.4f*sin(2t)]*u(t)\n', ...
        A, B, E, F/2);

% Alternative representation using magnitude and phase
R = sqrt(E^2 + (F/2)^2);  % Magnitude
phi = atan2(F/2, E);      % Phase angle (radians)
fprintf('Alternative form: y(t) = %.4f*t + %.4f + %.4f*exp(-t)*cos(2t + %.3f)\n', ...
        A, B, R, phi);

fprintf('\nNote: The presence of integrator (pole at s=0) means:\n');
fprintf('• Ramp term (t) in step response\n');
fprintf('• Zero steady-state error for step input\n');
fprintf('• Type 1 system behavior\n');

%% Method 4: System Characteristics Analysis - Integrator and Complex Roots
fprintf('\nMethod 4: System Characteristics - Integrator and Complex Roots\n');
fprintf('--------------------------------------------------------------\n');

% Find poles (roots of denominator)
poles_sym = solve(den_sym == 0, s);
fprintf('Symbolic poles: ');
for i = 1:length(poles_sym)
    fprintf('s%d = %s  ', i, char(poles_sym(i)));
end
fprintf('\n');

% For numerical example with integrator and complex roots
den_coeffs = [1, a2_val, a1_val, a0_val];  % [1, 2, 5, 0]
poles_num = roots(den_coeffs);
fprintf('Numerical poles:\n');
for i = 1:length(poles_num)
    if abs(poles_num(i)) < 1e-6
        fprintf('  s%d = 0.000 (integrator pole)\n', i);
    elseif abs(imag(poles_num(i))) < 1e-6
        fprintf('  s%d = %.3f (real)\n', i, real(poles_num(i)));
    else
        fprintf('  s%d = %.3f %+.3fj\n', i, real(poles_num(i)), imag(poles_num(i)));
    end
end

% Classify poles
real_poles = poles_num(abs(imag(poles_num)) < 1e-6);
complex_poles = poles_num(abs(imag(poles_num)) >= 1e-6);

fprintf('\nPole classification:\n');
fprintf('Real poles: %d (including integrator at origin)\n', length(real_poles));
fprintf('Complex poles: %d (in %d conjugate pairs)\n', length(complex_poles), length(complex_poles)/2);

% Check for integrator
integrator_poles = sum(abs(real_poles) < 1e-6);
fprintf('Integrator poles: %d\n', integrator_poles);
fprintf('System type: Type %d (number of integrators)\n', integrator_poles);

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

% Final value theorem for integrator system
fprintf('\nSteady-state analysis:\n');
if integrator_poles > 0
    fprintf('Final value for step input: y(∞) = ∞ (ramp response due to integrator)\n');
    fprintf('Steady-state error for step input: ess = 0 (Type %d system)\n', integrator_poles);
    
    % For ramp input, the steady-state error would be finite
    Kv = limit(s * G_num, s, 0);  % Velocity error constant
    fprintf('Velocity error constant Kv = %.3f\n', double(Kv));
else
    final_value = limit(s * Y_s_num, s, 0);
    fprintf('Final value (steady-state): y(∞) = %s = %.3f\n', ...
            char(final_value), double(final_value));
end

%% Method 5: Time Response Plotting
fprintf('\nMethod 5: Graphical Analysis\n');
fprintf('---------------------------\n');

% Convert symbolic expression to numerical function
t_vec = 0:0.01:3;  % Shorter time range for integrator system
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
title('Step Response - Type 1 System (Integrator)', 'FontSize', 14);
hold on;

% Add annotation about ramp component
A_ramp = 2;      % Ramp coefficient from partial fraction
B_step = -0.8;   % Step coefficient from partial fraction
ramp_line = A_ramp * t_vec + B_step;  
plot(t_vec, ramp_line, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('Ramp Component (%.1ft %+.1f)', A_ramp, B_step));
legend('Total Response', 'Ramp Component', 'Location', 'northwest');

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

% Subplot 3: Individual components for integrator and complex roots
subplot(2,2,3);
hold on;

% Use the calculated residues for integrator + complex roots
A = 2;           % 2.0 (ramp component from integrator)
B = -0.8;        % -0.8 (step component)
E = -2;          % -2.0 (cosine component)  
F = 6.8;         % 6.8 (sine component, adjusted F = D - C)

mode1 = A * t_vec;                                   % Ramp component (integrator effect)
mode2 = B * ones(size(t_vec));                       % Step component
mode3 = E * exp(-1*t_vec) .* cos(2*t_vec);           % Cosine × exponential
mode4 = (F/2) * exp(-1*t_vec) .* sin(2*t_vec);      % Sine × exponential
mode_complex = mode3 + mode4;                        % Combined oscillatory

plot(t_vec, mode1, 'r:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅t (ramp)', A));
plot(t_vec, mode2, 'g:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f (step)', B));
plot(t_vec, mode3, 'b:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅e^{-t}cos(2t)', E));
plot(t_vec, mode4, 'm:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅e^{-t}sin(2t)', F/2));
plot(t_vec, mode_complex, 'c-', 'LineWidth', 1.5, 'DisplayName', 'Combined Oscillatory');
plot(t_vec, y_values, 'k-', 'LineWidth', 2, 'DisplayName', 'Total Response');

grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Integrator + Complex Roots Decomposition', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);

% Subplot 4: System information for integrator and complex roots
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

% Calculate velocity error constant
Kv_val = K_val / a1_val;  % For Type 1 system: Kv = K/a1

info_text = {
    'System Information (Integrator + Complex):';
    '';
    sprintf('Transfer Function: G(s) = %d/(s(s²+2s+5))', K_val);
    sprintf('Expanded: G(s) = %d/(s³+%ds²+%ds+%d)', K_val, a2_val, a1_val, a0_val);
    '';
    'Poles:';
    sprintf('  s₁ = 0.000 (integrator)');
    sprintf('  s₂,₃ = %.1f ± j%.1f (complex)', sigma_display, omega_d_display);
    '';
    'Complex Pole Characteristics:';
    sprintf('  ωₙ = %.3f rad/s', omega_n_display);
    sprintf('  ζ = %.3f (underdamped)', zeta_display);
    sprintf('  ωd = %.3f rad/s', omega_d_display);
    '';
    'System Classification:';
    '  Type 1 (one integrator)';
    '  Order: 3rd';
    '  Zeros: None';
    sprintf('  Kv = %.1f', Kv_val);
    '';
    'Step Response Features:';
    '• Ramp component (from integrator)';
    '• Zero steady-state error';
    '• Damped oscillation';
    '• Growing amplitude over time';
};
text(0.05, 0.95, info_text, 'FontSize', 9, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');

sgtitle('3rd Order Plant with Integrator and Complex Roots - Complete Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% Method 6: Performance Metrics for Integrator System
fprintf('Method 6: Performance Metrics - Type 1 System with Integrator\n');
fprintf('-------------------------------------------------------------\n');

% Note about integrator system characteristics
fprintf('Note: This is a Type 1 system with integrator pole at s=0\n');
fprintf('Step response contains ramp component and grows without bound.\n\n');

% Since the response grows indefinitely due to integrator, 
% traditional metrics like settling time don't apply
fprintf('Traditional settling time: Not applicable (unbounded response)\n');

% Analyze the oscillatory component envelope
% The oscillatory part decays as exp(-t), so we can find when it becomes negligible
oscillation_envelope = abs(2 * exp(-1*t_vec));  % Approximate envelope magnitude
envelope_threshold = 0.02 * max(oscillation_envelope);  % 2% of max oscillation

envelope_settling_idx = find(oscillation_envelope <= envelope_threshold, 1);
if ~isempty(envelope_settling_idx)
    oscillation_settling_time = t_vec(envelope_settling_idx);
    fprintf('Oscillation settling time (envelope): %.3f seconds\n', oscillation_settling_time);
end

% Find first crossing of the ramp component (ignoring oscillations)
% Approximate ramp component value
ramp_component = A * t_vec + B;  % 2*t - 0.8
ramp_cross_1 = find(ramp_component >= 1, 1);  % When ramp reaches 1
if ~isempty(ramp_cross_1)
    fprintf('Time to reach unit amplitude (ramp): %.3f seconds\n', t_vec(ramp_cross_1));
end

% Peak analysis for oscillatory behavior
% Find local maxima and minima to analyze oscillation
[peaks, peak_locs] = findpeaks(y_values, 'MinPeakDistance', 20);
[valleys, valley_locs] = findpeaks(-y_values, 'MinPeakDistance', 20);
valleys = -valleys;

if length(peak_locs) >= 2
    % Calculate oscillation period from peaks
    oscillation_period_measured = mean(diff(t_vec(peak_locs))) * 2;  % Full period
    fprintf('Measured oscillation period: %.3f seconds\n', oscillation_period_measured);
    fprintf('Measured oscillation frequency: %.3f rad/s\n', 2*pi/oscillation_period_measured);
end

if length(peaks) >= 2
    % Calculate damping from successive peaks
    peak_ratio = peaks(2) / peaks(1);  % Note: not traditional damping due to ramp
    fprintf('Peak amplitude growth factor: %.3f (due to ramp + decay)\n', peak_ratio);
end

% Oscillation frequency (theoretical)
if ~isempty(complex_poles_plot)
    complex_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
    damped_freq = imag(complex_pole(1));
    oscillation_period = 2*pi / damped_freq;
    fprintf('Theoretical oscillation frequency: %.3f rad/s\n', damped_freq);
    fprintf('Theoretical oscillation period: %.3f seconds\n', oscillation_period);
end

% Velocity error constant (important for Type 1 systems)
Kv = K_val / a1_val;
fprintf('\nVelocity error constant Kv = %.3f\n', Kv);
fprintf('Steady-state error for ramp input: ess = 1/Kv = %.3f\n', 1/Kv);

fprintf('\nAnalysis complete! This Type 1 system shows:\n');
fprintf('• Zero steady-state error for step inputs\n');
fprintf('• Finite steady-state error for ramp inputs\n');
fprintf('• Unbounded step response due to integrator\n');
fprintf('• Superimposed damped oscillation\n');

%% Function for analyzing systems with integrator and complex roots
function analyze_integrator_complex_system(K, complex_sigma, complex_omega)
    % Helper function to analyze 3rd order systems with integrator and complex roots
    % Example: analyze_integrator_complex_system(10, -1, 2) 
    % Integrator at s=0, complex roots at complex_sigma ± j*complex_omega
    
    syms s t
    
    % Construct transfer function
    % G(s) = K/(s * ((s - complex_sigma)^2 + complex_omega^2))
    integrator_factor = s;
    complex_factor = (s - complex_sigma)^2 + complex_omega^2;
    G = K / (integrator_factor * complex_factor);
    
    fprintf('Integrator + complex roots system analysis:\n');
    fprintf('Integrator: s₁ = 0\n');
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
    
    % System type classification
    fprintf('\nSystem classification: Type 1 (one integrator)\n');
    
    % Error constants
    Kv = K / (complex_sigma^2 + complex_omega^2);  % Velocity error constant
    fprintf('Velocity error constant Kv = %.3f\n', Kv);
    
    fprintf('\nTime response characteristics:\n');
    fprintf('- Ramp component: A*t (from integrator)\n');
    fprintf('- Step component: B (from integrator)\n');
    fprintf('- Oscillatory terms: exp(%.1f*t)*[C*cos(%.1f*t) + D*sin(%.1f*t)]\n', ...
            complex_sigma, complex_omega);
    
    fprintf('\nSteady-state error characteristics:\n');
    fprintf('- Step input: ess = 0 (Type 1 system)\n');
    fprintf('- Ramp input: ess = 1/Kv = %.3f\n', 1/Kv);
    fprintf('- Parabolic input: ess = ∞ (Type 1 system)\n');
end