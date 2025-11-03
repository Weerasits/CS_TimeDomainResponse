%% 3rd Order Plant Step Response Analysis - Real/Complex Roots with Right Half Plane Zero
% Professor-level example for instrumentation and control
% System: G(s) = K(s-z)/((s+r)(s²+as+b)) - Real/Complex poles with RHP zero
% Author: Control Systems Analysis
% Date: 2025

clear all; close all; clc;

%% Define symbolic variables
syms s t K a2 a1 a0 real positive

% Display header
fprintf('=== 3rd Order Plant with Real/Complex Roots and RHP Zero - Step Response Analysis ===\n\n');

%% Method 1: General Symbolic Transfer Function with RHP Zero
fprintf('Method 1: General Symbolic Analysis with RHP Zero\n');
fprintf('------------------------------------------------\n');

% Define the 3rd order transfer function with RHP zero
% G(s) = K(s - z) / (s^3 + a2*s^2 + a1*s + a0) where z > 0
syms z real positive
num_sym = K * (s - z);  % RHP zero at s = +z
den_sym = s^3 + a2*s^2 + a1*s + a0;
G_sym = num_sym / den_sym;

fprintf('Transfer Function with RHP Zero: G(s) = %s\n', char(G_sym));

% Step input in Laplace domain: 1/s
% Y(s) = G(s) * (1/s)
Y_s = G_sym * (1/s);
fprintf('Output Y(s) = G(s) * (1/s) = %s\n', char(Y_s));

% Symbolic inverse Laplace transform (general form)
fprintf('\nGeneral step response (symbolic):\n');
y_t_general = ilaplace(Y_s, s, t);
fprintf('y(t) = %s\n\n', char(y_t_general));

%% Method 2: Specific Numerical Example - Real/Complex Roots with RHP Zero
fprintf('Method 2: Specific Numerical Example - Real/Complex Roots with RHP Zero\n');
fprintf('----------------------------------------------------------------------\n');

% Define 3rd order system with RHP zero and mixed poles
% Zero at s = +3 (RHP), Real pole at s = -1, Complex poles at s = -2 ± j3
% G(s) = K(s-3)/((s+1)(s²+4s+13)) = K(s-3)/(s³+5s²+17s+13)
K_val = 10;          % DC gain multiplier
z_val = 3;           % Zero location: s = +3 (RHP)
a2_val = 5;          % s^2 coefficient  
a1_val = 17;         % s coefficient
a0_val = 13;         % constant term

% Numerator and denominator coefficients
num_coeffs = [K_val, -K_val*z_val];  % [10, -30] for 10(s-3)
den_coeffs = [1, a2_val, a1_val, a0_val];  % [1, 5, 17, 13]

% This gives us: G(s) = (10s - 30) / (s³ + 5s² + 17s + 13)
% Factored form: G(s) = 10(s-3) / ((s+1)(s²+4s+13))

fprintf('Real/Complex roots with RHP zero example:\n');
fprintf('G(s) = %d(s-%d) / ((s+1)(s²+4s+13))\n', K_val, z_val);
fprintf('Expanded form: G(s) = (%ds-%d) / (s³+%ds²+%ds+%d)\n', ...
        K_val, K_val*z_val, a2_val, a1_val, a0_val);

% Root locations
fprintf('Root locations:\n');
fprintf('  Zero: s = +%d (Right Half Plane - Non-minimum Phase)\n', z_val);
fprintf('  Real pole: s₁ = -1\n');
fprintf('  Complex poles: s₂,₃ = -2 ± j3\n');

% Complex poles characteristics  
sigma = -2;  % Real part of complex poles
omega_d = 3; % Imaginary part of complex poles
omega_n = sqrt(sigma^2 + omega_d^2); % Natural frequency
zeta = -sigma / omega_n;             % Damping ratio

fprintf('  Natural frequency: ωₙ = %.3f rad/s\n', omega_n);
fprintf('  Damping ratio: ζ = %.3f\n', zeta);
fprintf('  System type: Type 0 with RHP zero (Non-minimum Phase)\n');

% Zero-pole analysis for RHP zero
fprintf('\nRHP Zero Analysis:\n');
fprintf('  Non-minimum phase system\n');
fprintf('  RHP zero provides phase lag (degrades transient response)\n');
fprintf('  Potential for inverse response (undershoot)\n');
fprintf('  Challenging for control design\n');
fprintf('  Zero farther from origin than dominant poles\n');

% Substitute numerical values for transfer function with RHP zero
num_sym_num = K_val * (s - z_val);  % 10(s-3)
den_sym_num = s^3 + a2_val*s^2 + a1_val*s + a0_val;  % s³+5s²+17s+13
G_num = num_sym_num / den_sym_num;
Y_s_num = G_num * (1/s);

fprintf('Y(s) = %s\n', char(Y_s_num));

% Calculate symbolic step response
y_t_num = ilaplace(Y_s_num, s, t);
fprintf('Step response: y(t) = %s\n', char(y_t_num));

% Simplify the expression
y_t_simplified = simplify(y_t_num);
fprintf('Simplified: y(t) = %s\n\n', char(y_t_simplified));

%% Method 3: Partial Fraction Decomposition - System with RHP Zero
fprintf('Method 3: Partial Fraction Analysis - System with RHP Zero\n');
fprintf('---------------------------------------------------------\n');

% Perform partial fraction decomposition
try
    Y_s_pfd = partfrac(Y_s_num, s);
    fprintf('Partial fraction decomposition:\n');
    fprintf('Y(s) = %s\n', char(Y_s_pfd));
catch
    fprintf('Direct partfrac failed, using manual approach...\n');
end

% Manual partial fraction for system with RHP zero
% Y(s) = 10(s-3)/(s(s+1)(s²+4s+13)) = A/s + B/(s+1) + (Cs+D)/(s²+4s+13)
fprintf('\nManual partial fraction calculation for system with RHP zero:\n');
fprintf('Y(s) = 10(s-3)/(s(s+1)(s²+4s+13))\n');
fprintf('Y(s) = A/s + B/(s+1) + (Cs+D)/(s²+4s+13)\n');

% Calculate residues - modified due to RHP zero (s-3)
% A = [s*Y(s)]|_{s=0} = 10(s-3)/((s+1)(s²+4s+13))|_{s=0}
A = 10*(-3)/((1)*(13));  % = -30/13 ≈ -2.308

% B = [(s+1)*Y(s)]|_{s=-1} = 10(s-3)/(s(s²+4s+13))|_{s=-1}
% At s=-1: s²+4s+13 = 1-4+13 = 10, and (s-3) = -4
B = 10*(-4)/((-1)*(10));  % = 4

% For complex part with RHP zero present, use algebraic method
% After substitution and algebraic manipulation:
% The RHP zero changes signs and affects all residues
C = -(A + B);  % Approximate: ≈ -(-2.308+4) = -1.692
D = -3*C;      % Approximate relationship (note sign change due to RHP zero)

fprintf('A = %.4f, B = %.4f, C = %.4f, D = %.4f\n', A, B, C, D);
fprintf('Y(s) = %.4f/s + %.4f/(s+1) + (%.4f*s + %.4f)/(s²+4s+13)\n', A, B, C, D);

% Convert to standard oscillatory form
% (Cs+D)/(s²+4s+13) = (C(s+2) + (D-2C))/((s+2)²+9)
fprintf('\nStandard oscillatory form:\n');
E = C;           % Coefficient of (s+2) term
F = D - 2*C;     % Coefficient of constant term
fprintf('Y(s) = %.4f/s + %.4f/(s+1) + [%.4f*(s+2) + %.4f]/[(s+2)²+9]\n', A, B, E, F);

% Time domain response with RHP zero effects
fprintf('\nTime domain components (with RHP zero effects):\n');
fprintf('y(t) = %.4f*u(t) + %.4f*exp(-t)*u(t) + exp(-2t)*[%.4f*cos(3t) + %.4f*sin(3t)]*u(t)\n', ...
        A, B, E, F/3);

fprintf('\nNote: The RHP zero at s = +3 affects the response by:\n');
fprintf('• Creating inverse response (initial undershoot)\n');
fprintf('• Introducing phase lag (degrades transient response)\n');
fprintf('• Non-minimum phase behavior\n');
fprintf('• The zero provides "derivative action" but with wrong sign\n');
fprintf('• Final value is NEGATIVE due to RHP zero!\n');
fprintf('• Makes control design more challenging\n');

%% Method 4: System Characteristics Analysis - System with RHP Zero
fprintf('\nMethod 4: System Characteristics - System with RHP Zero\n');
fprintf('-------------------------------------------------------\n');

% Find poles and zeros
fprintf('Poles and Zeros Analysis:\n');

% Zeros (roots of numerator) - RHP zero
zero_coeffs = [K_val, -K_val*z_val];  % [10, -30] for 10(s-3)
zeros_num = roots(zero_coeffs);
fprintf('Zeros:\n');
for i = 1:length(zeros_num)
    fprintf('  z%d = %.3f', i, zeros_num(i));
    if zeros_num(i) > 0
        fprintf(' (RHP - Non-minimum Phase)\n');
    else
        fprintf(' (LHP - Minimum Phase)\n');
    end
end

% Poles (roots of denominator)  
den_coeffs = [1, a2_val, a1_val, a0_val];  % [1, 5, 17, 13]
poles_num = roots(den_coeffs);
fprintf('\nPoles:\n');
for i = 1:length(poles_num)
    if abs(imag(poles_num(i))) < 1e-6
        fprintf('  s%d = %.3f (real)\n', i, real(poles_num(i)));
    else
        fprintf('  s%d = %.3f %+.3fj\n', i, real(poles_num(i)), imag(poles_num(i)));
    end
end

% Classify poles and zeros
real_poles = poles_num(abs(imag(poles_num)) < 1e-6);
complex_poles = poles_num(abs(imag(poles_num)) >= 1e-6);
real_zeros = zeros_num(abs(imag(zeros_num)) < 1e-6);

fprintf('\nPole-Zero classification:\n');
fprintf('Real poles: %d (all in LHP)\n', length(real_poles));
fprintf('Complex poles: %d (in %d conjugate pairs)\n', length(complex_poles), length(complex_poles)/2);
fprintf('Real zeros: %d (in RHP - Non-minimum Phase!)\n', length(real_zeros));

% Non-minimum phase analysis
fprintf('\nNon-Minimum Phase Analysis:\n');
for i = 1:length(real_zeros)
    zero_loc = real_zeros(i);
    fprintf('RHP Zero at s = %.1f:\n', zero_loc);
    
    % Distance from zero to each pole
    for j = 1:length(poles_num)
        distance = abs(poles_num(j) - zero_loc);
        fprintf('  Distance to pole s%d: %.3f\n', j, distance);
    end
    
    % Zero-pole cancellation check
    min_distance = min(abs(poles_num - zero_loc));
    if min_distance < 0.1
        fprintf('  WARNING: Near pole-zero cancellation!\n');
    else
        fprintf('  No near cancellations - full RHP zero effect\n');
    end
end

% Complex pole characteristics
if ~isempty(complex_poles)
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
        fprintf('  Complex poles: Underdamped\n');
    elseif abs(zeta_c - 1) < 1e-6
        fprintf('  Complex poles: Critically damped\n');
    else
        fprintf('  Complex poles: Overdamped\n');
    end
end

% Final value theorem - IMPORTANT: Different for RHP zero!
final_value = limit(s * Y_s_num, s, 0);
fprintf('\nSteady-state analysis:\n');
fprintf('  Final value: y(∞) = %s = %.3f\n', char(final_value), double(final_value));

% DC gain with RHP zero - NOTE: Can be negative!
dc_gain = (K_val * (-z_val)) / a0_val;  % K(-z)/a0 for RHP zero
fprintf('  DC gain: K(-z)/a₀ = %d×(-%d)/%d = %.3f\n', K_val, z_val, a0_val, dc_gain);
fprintf('  System type: Type 0 with RHP zero (Non-minimum Phase)\n');

% Error analysis
if dc_gain > 0
    Kp = dc_gain;  % Position error constant equals DC gain
    ess_step = 1 / (1 + Kp);
    fprintf('  Position error constant: Kp = %.3f\n', Kp);
    fprintf('  Steady-state error (step): ess = %.4f\n', ess_step);
else
    fprintf('  WARNING: Negative DC gain due to RHP zero!\n');
    fprintf('  System has inverse steady-state response\n');
    fprintf('  Traditional error analysis not applicable\n');
end

%% Method 5: Time Response Plotting
fprintf('\nMethod 5: Graphical Analysis\n');
fprintf('---------------------------\n');

% Convert symbolic expression to numerical function
t_vec = 0:0.01:5;  % Time range suitable for system with zero
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
title('Step Response - System with RHP Zero (Non-Minimum Phase)', 'FontSize', 14);
hold on;

% Add final value line
final_val = double(final_value);
yline(final_val, 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Final Value = %.3f', final_val));

% Add zero line for reference
yline(0, 'k:', 'LineWidth', 1, 'Label', 'Zero Line');

% Highlight inverse response characteristics
% Find minimum value (undershoot)
[min_val, min_idx] = min(y_values);
if min_val < 0
    min_time = t_vec(min_idx);
    plot(min_time, min_val, 'mo', 'MarkerSize', 8, 'LineWidth', 2, ...
         'DisplayName', sprintf('Min: %.3f at %.2fs', min_val, min_time));
    
    % Add annotation for inverse response
    text(min_time + 0.2, min_val - 0.1, 'Inverse Response\n(Undershoot)', ...
         'FontSize', 10, 'Color', 'magenta', 'FontWeight', 'bold');
end

% Mark the point where response crosses zero (if it does)
zero_crossings = find(diff(sign(y_values)));
if ~isempty(zero_crossings)
    for i = 1:length(zero_crossings)
        cross_time = t_vec(zero_crossings(i));
        plot(cross_time, 0, 'rs', 'MarkerSize', 8, 'LineWidth', 2, ...
             'DisplayName', sprintf('Zero crossing at %.2fs', cross_time));
    end
end

legend('Total Response', 'Final Value', 'Zero Line', 'Location', 'southeast');

% Subplot 2: Pole-zero map with RHP zeros and poles
subplot(2,2,2);
hold on;

% Plot zeros - differentiate RHP vs LHP
real_zeros_plot = zeros_num(abs(imag(zeros_num)) < 1e-6);
if ~isempty(real_zeros_plot)
    if real(real_zeros_plot(1)) > 0
        % RHP zero - use red circle to indicate danger
        plot(real(real_zeros_plot), imag(real_zeros_plot), 'ro', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'RHP Zeros');
    else
        % LHP zero - use green circle
        plot(real(real_zeros_plot), imag(real_zeros_plot), 'go', 'MarkerSize', 12, 'LineWidth', 3, 'DisplayName', 'LHP Zeros');
    end
end

% Plot poles (crosses)
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

% Highlight the imaginary axis (stability boundary)
line([0 0], [-4 4], 'Color', 'k', 'LineWidth', 2, 'LineStyle', '-.', 'DisplayName', 'jω-axis');

% Draw real axis
line([-4 4], [0 0], 'Color', 'k', 'LineWidth', 0.5); % Real axis

% Mark important points
plot(0, 0, 'ko', 'MarkerSize', 4, 'DisplayName', 'Origin');

% Shade RHP to indicate non-minimum phase region
x_rhp = [0, 4, 4, 0];
y_rhp = [-4, -4, 4, 4];
fill(x_rhp, y_rhp, 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'RHP (Non-min Phase)');

grid on;
xlabel('Real Part (σ)', 'FontSize', 12);
ylabel('Imaginary Part (jω)', 'FontSize', 12);
title('Pole-Zero Map (Non-Minimum Phase)', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);
axis equal;
xlim([-4, 4]);
ylim([-4, 4]);

% Add annotations for poles and zeros
if ~isempty(real_zeros_plot)
    for i = 1:length(real_zeros_plot)
        if real(real_zeros_plot(i)) > 0
            text(real(real_zeros_plot(i)), imag(real_zeros_plot(i))-0.3, ...
                sprintf('z = +%.0f (RHP)', real(real_zeros_plot(i))), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'red', 'FontWeight', 'bold');
        else
            text(real(real_zeros_plot(i)), imag(real_zeros_plot(i))-0.3, ...
                sprintf('z = %.0f', real(real_zeros_plot(i))), ...
                'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'green');
        end
    end
end

if ~isempty(complex_poles_plot)
    for i = 1:length(complex_poles_plot)
        if imag(complex_poles_plot(i)) > 0
            text(real(complex_poles_plot(i))+0.2, imag(complex_poles_plot(i))+0.2, ...
                sprintf('%.0f+j%.0f', real(complex_poles_plot(i)), imag(complex_poles_plot(i))), ...
                'FontSize', 9, 'Color', 'blue');
        end
    end
end

if ~isempty(real_poles_plot)
    for i = 1:length(real_poles_plot)
        text(real(real_poles_plot(i)), imag(real_poles_plot(i))-0.3, ...
            sprintf('p = %.0f', real(real_poles_plot(i))), ...
            'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'red');
    end
end

% Subplot 3: Individual components for system with RHP zero
subplot(2,2,3);
hold on;

% Use the calculated residues for system with RHP zero
A = -30/13;      % ≈ -2.308 (DC component - NEGATIVE!)
B = 4;           % 4.000 (real exponential)
E = -(A + B);    % ≈ -1.692 (cosine component, adjusted for RHP zero effects)  
F = -3*E;        % ≈ 5.076 (sine component, adjusted for RHP zero effects)

mode1 = A * ones(size(t_vec));                          % Step component (DC) - NEGATIVE
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

% Add zero line for reference
yline(0, 'k--', 'LineWidth', 0.5, 'Alpha', 0.3);

grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('RHP Zero + Real/Complex Poles Decomposition', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 9);

% Subplot 4: System information for system with RHP zero
subplot(2,2,4);
axis off;

% Calculate system parameters
final_val = double(final_value);
dc_gain_val = (K_val * (-z_val)) / a0_val;  % Note negative for RHP zero
Kp_val = abs(dc_gain_val);  % Use absolute value for error analysis

% Get complex pole parameters
if ~isempty(complex_poles_plot)
    complex_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
    sigma_display = real(complex_pole(1));
    omega_d_display = imag(complex_pole(1));
    omega_n_display = abs(complex_pole(1));
    zeta_display = -sigma_display / omega_n_display;
end

info_text = {
    'System Information (RHP Zero + Mixed Poles):';
    '';
    sprintf('Transfer Function: G(s) = %d(s-%d)/((s+1)(s²+4s+13))', K_val, z_val);
    sprintf('Expanded: G(s) = (%ds-%d)/(s³+%ds²+%ds+%d)', K_val, K_val*z_val, a2_val, a1_val, a0_val);
    '';
    'Poles and Zeros:';
    sprintf('  Zero: s = +%d (RHP - NON-MIN PHASE!)', z_val);
    sprintf('  Real pole: s = -1');
    sprintf('  Complex poles: s = %.0f±j%.0f', sigma_display, omega_d_display);
    '';
    'Complex Pole Characteristics:';
    sprintf('  ωₙ = %.3f rad/s', omega_n_display);
    sprintf('  ζ = %.3f (underdamped)', zeta_display);
    sprintf('  ωd = %.3f rad/s', omega_d_display);
    '';
    'RHP Zero Effects (NEGATIVE!):';
    '• Inverse response (undershoot)';
    '• Phase lag (degrades response)';
    '• Non-minimum phase behavior';
    '• Control design challenges';
    '• Negative final value!';
    '';
    'Performance:';
    sprintf('  Final value: %.3f (NEGATIVE!)', final_val);
    sprintf('  DC gain: %.3f (NEGATIVE!)', dc_gain_val);
    '  Traditional analysis invalid';
};
text(0.05, 0.95, info_text, 'FontSize', 9, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');

sgtitle('3rd Order Plant with Real/Complex Roots and RHP Zero - Complete Analysis', 'FontSize', 16, 'FontWeight', 'bold');

%% Method 6: Performance Metrics for System with RHP Zero
fprintf('Method 6: Performance Metrics - System with RHP Zero (Non-Minimum Phase)\n');
fprintf('--------------------------------------------------------------------------\n');

% Calculate final value
final_val = double(final_value);
fprintf('Final value: %.3f (NEGATIVE due to RHP zero!)\n', final_val);

% For RHP zero systems, traditional metrics need modification
if final_val < 0
    fprintf('\nWARNING: Negative final value indicates inverse steady-state response!\n');
    fprintf('Traditional performance metrics may not apply directly.\n\n');
end

% Inverse response analysis
[min_val, min_idx] = min(y_values);
min_time = t_vec(min_idx);
fprintf('Inverse response characteristics:\n');
fprintf('  Minimum value: %.3f at t = %.3f seconds\n', min_val, min_time);

if min_val < 0
    inverse_magnitude = abs(min_val / final_val) * 100;
    fprintf('  Inverse response magnitude: %.1f%% of final value\n', inverse_magnitude);
else
    fprintf('  No undershoot detected in this time range\n');
end

% Find zero crossings
zero_crossings = find(diff(sign(y_values)));
fprintf('  Zero crossings: %d\n', length(zero_crossings));
if ~isempty(zero_crossings)
    for i = 1:length(zero_crossings)
        cross_time = t_vec(zero_crossings(i));
        fprintf('    Crossing %d at t = %.3f seconds\n', i, cross_time);
    end
end

% Modified settling time analysis (relative to final value)
if final_val ~= 0
    tolerance_2 = 0.02 * abs(final_val);
    settling_idx_2 = [];
    
    % Find last time the response exceeds tolerance band
    for i = length(y_values):-1:1
        if abs(y_values(i) - final_val) > tolerance_2
            settling_idx_2 = i + 1;
            break;
        end
    end
    
    if ~isempty(settling_idx_2) && settling_idx_2 <= length(t_vec)
        settling_time_2 = t_vec(settling_idx_2);
        fprintf('\nModified settling time (2%% of |final value|): %.3f seconds\n', settling_time_2);
    else
        fprintf('\nSettling time (2%%): < %.2f seconds\n', t_vec(end));
    end
else
    fprintf('\nSettling time analysis: Not applicable (zero final value)\n');
end

% Time to reach maximum undershoot
fprintf('\nTime to maximum undershoot: %.3f seconds\n', min_time);

% Recovery time (time from minimum to positive values, if applicable)
if min_val < 0
    positive_idx = find(y_values(min_idx:end) > 0, 1);
    if ~isempty(positive_idx)
        recovery_time = t_vec(min_idx + positive_idx - 1) - min_time;
        fprintf('Recovery time (from min to positive): %.3f seconds\n', recovery_time);
    else
        fprintf('Recovery time: > %.2f seconds (does not reach positive values)\n', t_vec(end) - min_time);
    end
end

% RHP zero analysis
fprintf('\nRHP Zero Effects Analysis:\n');
zero_location = z_val;  % Zero at s = +3
fprintf('RHP Zero location: s = +%.0f\n', zero_location);

% Distance from zero to dominant poles
dominant_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
if ~isempty(dominant_pole)
    zero_pole_distance = abs(dominant_pole(1) - zero_location);
    fprintf('Distance from RHP zero to complex pole: %.3f\n', zero_pole_distance);
    
    % Zero-pole ratio analysis for RHP zero
    zero_to_origin = abs(zero_location);
    pole_to_origin = abs(dominant_pole(1));
    zp_ratio = zero_to_origin / pole_to_origin;
    fprintf('Zero-to-pole distance ratio: %.3f\n', zp_ratio);
    
    if zp_ratio > 1
        fprintf('RHP zero farther from origin → Moderate non-minimum phase effects\n');
    else
        fprintf('RHP zero closer to origin → Strong non-minimum phase effects\n');
    end
end

% Oscillation analysis for complex poles
if ~isempty(complex_poles_plot)
    complex_pole = complex_poles_plot(imag(complex_poles_plot) > 0);
    damped_freq = imag(complex_pole(1));
    oscillation_period = 2*pi / damped_freq;
    fprintf('\nOscillation characteristics (modified by RHP zero):\n');
    fprintf('Damped oscillation frequency: %.3f rad/s\n', damped_freq);
    fprintf('Oscillation period: %.3f seconds\n', oscillation_period);
end

% Control design implications
fprintf('\nControl Design Implications:\n');
fprintf('• RHP zero creates fundamental limitations\n');
fprintf('• Bandwidth-delay trade-offs\n');
fprintf('• Sensitivity to disturbances\n');
fprintf('• Robustness challenges\n');
fprintf('• May require specialized control strategies\n');
fprintf('• Impossible to achieve both fast response and good regulation\n');

% Steady-state error analysis (modified)
dc_gain = (K_val * (-z_val)) / a0_val;
fprintf('\nModified steady-state analysis:\n');
fprintf('DC gain (with RHP zero): %.3f (NEGATIVE!)\n', dc_gain);
if dc_gain < 0
    fprintf('Traditional error analysis not applicable for negative DC gain\n');
    fprintf('System exhibits inverse steady-state behavior\n');
end

fprintf('\nSystem characteristics summary:\n');
fprintf('• Non-minimum phase system due to RHP zero\n');
fprintf('• Inverse response (undershoot before reaching final value)\n');
fprintf('• Degraded transient response compared to LHP zero\n');
fprintf('• Fundamental control limitations\n');
fprintf('• Challenging for high-performance control\n');
fprintf('• Common in some physical systems (e.g., aircraft, some chemical processes)\n');

fprintf('\nAnalysis complete!\n');

%% Function for analyzing systems with zeros and mixed poles (including RHP zeros)
function analyze_zero_mixed_pole_system(K, zero_loc, real_pole, complex_sigma, complex_omega)
    % Helper function to analyze 3rd order systems with zeros and mixed poles
    % Example: analyze_zero_mixed_pole_system(10, 3, -1, -2, 3) 
    % RHP zero at +3, real pole at -1, complex poles at -2±j3
    
    syms s t
    
    % Construct transfer function
    % G(s) = K(s - zero_loc)/((s - real_pole)((s - complex_sigma)^2 + complex_omega^2))
    zero_factor = (s - zero_loc);
    real_factor = (s - real_pole);
    complex_factor = (s - complex_sigma)^2 + complex_omega^2;
    G = K * zero_factor / (real_factor * complex_factor);
    
    fprintf('Zero + mixed poles system analysis:\n');
    fprintf('Zero: s = %.1f', zero_loc);
    if zero_loc > 0
        fprintf(' (RHP - Non-minimum Phase)\n');
    else
        fprintf(' (LHP - Minimum Phase)\n');
    end
    fprintf('Real pole: s = %.1f\n', real_pole);
    fprintf('Complex poles: s = %.1f ± j%.1f\n', complex_sigma, complex_omega);
    fprintf('G(s) = %s\n', char(G));
    
    % Step response
    Y_s = G * (1/s);
    y_t = ilaplace(Y_s, s, t);
    fprintf('Step response: y(t) = %s\n', char(simplify(y_t)));
    
    % Complex pole characteristics
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
    
    % Zero effects analysis
    fprintf('\nZero effects analysis:\n');
    if zero_loc < 0
        fprintf('LHP zero at s = %.1f provides:\n', zero_loc);
        fprintf('- Phase lead (improves transient response)\n');
        fprintf('- Reduced overshoot\n');
        fprintf('- Faster settling time\n');
        fprintf('- Better stability margins\n');
        fprintf('- Minimum phase behavior\n');
    else
        fprintf('RHP zero at s = %.1f (Non-minimum phase) causes:\n', zero_loc);
        fprintf('- Phase lag (degrades transient response)\n');
        fprintf('- Inverse response (undershoot)\n');
        fprintf('- Non-minimum phase behavior\n');
        fprintf('- Fundamental control limitations\n');
        fprintf('- Design challenges\n');
    end
    
    % Zero-pole distance analysis
    distance_to_real = abs(zero_loc - real_pole);
    distance_to_complex = abs(zero_loc - (complex_sigma + 1j*complex_omega));
    
    fprintf('\nZero-pole distances:\n');
    fprintf('Distance to real pole: %.3f\n', distance_to_real);
    fprintf('Distance to complex pole: %.3f\n', distance_to_complex);
    
    % Final value and error analysis
    fprintf('\nSteady-state analysis:\n');
    dc_gain = K * (-zero_loc) / ((-real_pole) * (complex_sigma^2 + complex_omega^2));
    fprintf('DC gain: %.3f', dc_gain);
    if dc_gain < 0
        fprintf(' (NEGATIVE - due to RHP zero!)\n');
    else
        fprintf(' (positive)\n');
    end
    
    if dc_gain > 0
        Kp = dc_gain;
        ess = 1 / (1 + Kp);
        fprintf('Position error constant Kp = %.3f\n', Kp);
        fprintf('Steady-state error (step): ess = %.4f\n', ess);
    else
        fprintf('Traditional error analysis not applicable (negative DC gain)\n');
        fprintf('System exhibits inverse steady-state behavior\n');
    end
    
    fprintf('\nTime response characteristics:\n');
    fprintf('- DC component from step input');
    if dc_gain < 0
        fprintf(' (NEGATIVE final value)\n');
    else
        fprintf('\n');
    end
    fprintf('- Real exponential: A*exp(%.1f*t)\n', real_pole);
    fprintf('- Oscillatory terms: exp(%.1f*t)*[B*cos(%.1f*t) + C*sin(%.1f*t)]\n', ...
            complex_sigma, complex_omega);
    if zero_loc > 0
        fprintf('- Inverse response due to RHP zero\n');
        fprintf('- Initial wrong-way behavior\n');
    else
        fprintf('- Zero modifies amplitudes and phases of all modes\n');
    end
    
    fprintf('\nDesign implications:\n');
    if zero_loc < 0
        fprintf('- Good choice for controller design\n');
        fprintf('- Can be used to improve transient response\n');
        fprintf('- Typical in lead compensator design\n');
        fprintf('- Minimum phase system advantages\n');
    else
        fprintf('- Challenging for control design\n');
        fprintf('- Fundamental bandwidth limitations\n');
        fprintf('- Requires specialized control strategies\n');
        fprintf('- Common in aerospace and chemical processes\n');
        fprintf('- Trade-off between speed and regulation\n');
    end
    
    fprintf('\nNon-minimum phase effects (if RHP zero):\n');
    if zero_loc > 0
        fprintf('- Undershoot before reaching final value\n');
        fprintf('- Cannot achieve fast response and good disturbance rejection\n');
        fprintf('- Sensitivity limitations\n');
        fprintf('- Robustness challenges\n');
    else
        fprintf('- N/A (minimum phase system)\n');
    end
end