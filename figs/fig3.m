figure()

% So, first we fix an alpha
alpha = .3; 
L = 100; 
kappas = -linspace(.54, .6, 300);

% Define most of the ODE
diff = @(t, y) [y(2); y(3); y(1)*((-y(3))^(2-alpha))/(alpha*(alpha+1))];
ivs = [0; 1; 0];
dom = [0 L];
options = odeset('RelTol',1e-10,'AbsTol',1e-15);

for kappa = kappas
    ivs(3) = kappa;
    [t, y] = ode15s(diff, dom, ivs, options);
    y=real(y);
    [pks, locs] = findpeaks(y(:,1), t);
    kappa
    col1 = [0 0.5 0 .3];
    col = [0 0 0.5 .3];
     if ~isempty(pks)
        col=col1;
    end
    % Plot the result
    plot(t, y(:,1), 'Color', col, 'DisplayName', num2str(kappa)); hold on
    % if ~isempty(pks)
    %     plot(locs(1), pks(1), 'm*')
    % end
end

title(['Shooting Plot for alpha = ' num2str(alpha)])
saveas(gcf, 'fig3.png')
