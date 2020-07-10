clear all;

plot_shaded('epscores.mat', 10);
plot_shaded('losses_total.mat', 10);
plot_shaded('losses_ppo.mat', 10);
plot_shaded('losses_vest.mat', 10);


function plot_shaded(filepath, window)
    load(filepath);

    stdv = movstd(y, window);
    meanv = movmean(y, window);
    
    figure
    plot(x, y, 'b')
    plot(x, meanv, 'r--')
    hold on
    patch([x fliplr(x)], [y-stdv fliplr(y+stdv)], 'r', 'FaceAlpha', 0.2, 'EdgeColor','none')
    hold off
    title(regexprep(filepath,'_','','emptymatch'))
end