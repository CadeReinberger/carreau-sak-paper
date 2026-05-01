figure()

% So, first we fix an alpha
alpha = 1.4; 
L = 10; 
kappas = -linspace(.4, .5, 100);

% Define most of the ODE
diff = @(t, y) [y(2); y(3); y(1)*((-y(3))^(2-alpha))/(alpha*(alpha+1))];
ivs = [0; 1; 0];
dom = [0 L];
options = odeset('RelTol',1e-10,'AbsTol',1e-15);

n_kappas = length(kappas);
all_t = cell(n_kappas, 1);
all_y1 = cell(n_kappas, 1);
all_is_green = false(n_kappas, 1);

for idx = 1:n_kappas
    kappa = kappas(idx);
    ivs(3) = kappa;
    [t, y] = ode15s(diff, dom, ivs, options);
    y=real(y);
    [pks, locs] = findpeaks(y(:,1), t);
    kappa
    col1 = [0 0.5 0 .3];
    col = [0 0 0.5 .3];
    if ~isempty(pks)
        col=col1;
        all_is_green(idx) = true;
    end
    all_t{idx} = t;
    all_y1{idx} = y(:,1);
    % Plot the result
    plot(t, y(:,1), 'Color', col, 'DisplayName', num2str(kappa)); hold on
    % if ~isempty(pks)
    %     plot(locs(1), pks(1), 'm*')
    % end
end

% Find the boundary pair: last blue (largest-magnitude kappa still blue)
% and the consecutive green just past it, then plot their arithmetic mean.
blue_idxs = find(~all_is_green);
if ~isempty(blue_idxs)
    last_blue = max(blue_idxs);
    next_idx = last_blue + 1;
    if next_idx <= n_kappas && all_is_green(next_idx)
        t1 = all_t{last_blue};  y1 = all_y1{last_blue};
        t2 = all_t{next_idx};   y2 = all_y1{next_idx};
        t_end = min(t1(end), t2(end));
        tc = linspace(0, t_end, 2000);
        yc1 = interp1(t1, y1, tc);
        yc2 = interp1(t2, y2, tc);
        y_mean = (yc1 + yc2) / 2;
        plot(tc, y_mean, 'k--', 'LineWidth', 2);
    end
end

title(['Shooting Plot for alpha = ' num2str(alpha)])
saveas(gcf, 'fig6.png')
