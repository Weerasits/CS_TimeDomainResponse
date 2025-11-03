%% 3rd Order Plant Step Response Analysis - Symbolic Approach
% Professor-level example for instrumentation and control
% Author: Control Systems Analysis
% Date: 2025

clear all; close all; clc;

%% Define symbolic variables
syms s t K a2 a1 a0 real positive

% Display header
fprintf('=== 3rd Order Plant Step Response Analysis ===\n\n');

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

%% Method 2: Specific Numerical Example
fprintf('Method 2: Specific Numerical Example\n');
fprintf('------------------------------------\n');

% Define specific coefficients for a stable 3rd order system
K_val = 10;          % DC gain
a2_val = 6;          % s^2 coefficient  
a1_val = 11;         % s coefficient
a0_val = 6;          % constant term

% This gives us: G(s) = 10 / (s^3 + 6s^2 + 11s + 6)
% Factored form: G(s) = 10 / ((s+1)(s+2)(s+3))

fprintf('Specific example: G(s) = %d / (s^3 + %ds^2 + %ds + %d)\n', ...
        K_val, a2_val, a1_val, a0_val);

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

%% Method 3: Partial Fraction Decomposition
fprintf('Method 3: Partial Fraction Analysis\n');
fprintf('-----------------------------------\n');

% Perform partial fraction decomposition
try
    Y_s_pfd = partfrac(Y_s_num, s);
    fprintf('Partial fraction decomposition:\n');
    fprintf('Y(s) = %s\n', char(Y_s_pfd));
catch
    fprintf('Direct partfrac failed, using manual approach...\n');
end

% Manual partial fraction for the specific example
% Y(s) = 10/(s(s+1)(s+2)(s+3)) = A/s + B/(s+1) + C/(s+2) + D/(s+3)
fprintf('\nManual partial fraction calculation:\n');
fprintf('Y(s) = 10/(s(s+1)(s+2)(s+3))\n');

% Calculate residues manually using cover-up method
A = 10/((1)*(2)*(3));                    % Residue at s=0: 10/6
B = 10/((-1)*(-1+2)*(-1+3));            % Residue at s=-1: 10/(-2) = -5  
C = 10/((-2)*(-2+1)*(-2+3));            % Residue at s=-2: 10/2 = 5
D = 10/((-3)*(-3+1)*(-3+2));            % Residue at s=-3: 10/(-6) = -5/3

fprintf('A = %.4f, B = %.4f, C = %.4f, D = %.4f\n', A, B, C, D);
fprintf('Y(s) = %.4f/s + (%.4f)/(s+1) + %.4f/(s+2) + (%.4f)/(s+3)\n', A, B, C, D);

% Time domain response
fprintf('\nTime domain components:\n');
fprintf('y(t) = %.4f*u(t) + (%.4f)*exp(-t)*u(t) + %.4f*exp(-2t)*u(t) + (%.4f)*exp(-3t)*u(t)\n', ...
        A, B, C, D);

%% Method 4: System Characteristics Analysis
fprintf('\nMethod 4: System Characteristics\n');
fprintf('--------------------------------\n');

% Find poles (roots of denominator)
poles_sym = solve(den_sym == 0, s);
fprintf('Symbolic poles: ');
for i = 1:length(poles_sym)
    fprintf('s%d = %s  ', i, char(poles_sym(i)));
end
fprintf('\n');

% For numerical example
den_coeffs = [1, a2_val, a1_val, a0_val];
poles_num = roots(den_coeffs);
fprintf('Numerical poles: ');
for i = 1:length(poles_num)
    fprintf('s%d = %.3f  ', i, poles_num(i));
end
fprintf('\n');

% Final value theorem
final_value = limit(s * Y_s_num, s, 0);
fprintf('Final value (steady-state): y(∞) = %s = %.3f\n', ...
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

% Subplot 2: Pole-zero map
subplot(2,2,2);
plot(real(poles_num), imag(poles_num), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
grid on;
xlabel('Real Part', 'FontSize', 12);
ylabel('Imaginary Part', 'FontSize', 12);
title('Pole Locations', 'FontSize', 14);
axis equal;

% Subplot 3: Individual exponential components
subplot(2,2,3);
hold on;

% Use the correctly calculated residues
A = 10/6;        % 1.6667
B = -5;          % -5.0000
C = 5;           % 5.0000  
D = -10/6;       % -1.6667

mode1 = A * ones(size(t_vec));       % Step component (DC)
mode2 = B * exp(-1*t_vec);           % Pole at s=-1
mode3 = C * exp(-2*t_vec);           % Pole at s=-2  
mode4 = D * exp(-3*t_vec);           % Pole at s=-3

plot(t_vec, mode1, 'r:', 'LineWidth', 1.5, 'DisplayName', sprintf('DC = %.3f', A));
plot(t_vec, mode2, 'g:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅e^{-t}', B));
plot(t_vec, mode3, 'b:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.1f⋅e^{-2t}', C));
plot(t_vec, mode4, 'm:', 'LineWidth', 1.5, 'DisplayName', sprintf('%.3f⋅e^{-3t}', D));
plot(t_vec, y_values, 'k-', 'LineWidth', 2, 'DisplayName', 'Total Response');

grid on;
xlabel('Time (s)', 'FontSize', 12);
ylabel('Amplitude', 'FontSize', 12);
title('Exponential Mode Decomposition', 'FontSize', 14);
legend('Location', 'best');

% Subplot 4: System information
subplot(2,2,4);
axis off;
info_text = {
    'System Information:';
    '';
    sprintf('Transfer Function: G(s) = %d/(s³+%ds²+%ds+%d)', K_val, a2_val, a1_val, a0_val);
    '';
    'Poles:';
    sprintf('  s₁ = %.3f', poles_num(1));
    sprintf('  s₂ = %.3f', poles_num(2));
    sprintf('  s₃ = %.3f', poles_num(3));
    '';
    sprintf('Final Value: %.3f', double(final_value));
    '';
    'System Type: Stable (all poles in LHP)';
    'Order: 3rd';
    'Zeros: None';
};
text(0.1, 0.9, info_text, 'FontSize', 11, 'VerticalAlignment', 'top', ...
     'FontName', 'FixedWidth');

sgtitle('3rd Order Plant Analysis - Complete Overview', 'FontSize', 16, 'FontWeight', 'bold');

%% Method 6: Performance Metrics
fprintf('Method 6: Performance Metrics\n');
fprintf('-----------------------------\n');

% Calculate settling time (2% criterion)
final_val = double(final_value);
settling_idx = find(abs(y_values - final_val) <= 0.02 * final_val, 1);
if ~isempty(settling_idx)
    settling_time = t_vec(settling_idx);
    fprintf('Settling time (2%% criterion): %.3f seconds\n', settling_time);
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

fprintf('\nAnalysis complete!\n');

%% Function for further analysis
function analyze_custom_system(K, a2, a1, a0)
    % Helper function to analyze any 3rd order system
    syms s t
    G = K / (s^3 + a2*s^2 + a1*s + a0);
    Y_s = G * (1/s);
    y_t = ilaplace(Y_s, s, t);
    
    fprintf('Custom System Analysis:\n');
    fprintf('G(s) = %d / (s^3 + %ds^2 + %ds + %d)\n', K, a2, a1, a0);
    fprintf('Step response: y(t) = %s\n', char(simplify(y_t)));
end