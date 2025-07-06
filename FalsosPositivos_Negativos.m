classes = {'ceu_limpo', 'chama', 'chama_fumo', 'fumo', 'nuvem_neblina'};
TP = [40, 33, 60, 65, 39];
FP = [3, 5, 8, 6, 2];
FN = [3, 14, 25, 9, 4];

x = 1:numel(classes);
barWidth = 0.25;

figure;
hold on;
bar(x - barWidth, TP, barWidth, 'FaceColor', [0.2 0.7 0.2]); % TP - verde
bar(x, FP, barWidth, 'FaceColor', [0.9 0.2 0.2]);             % FP - vermelho
bar(x + barWidth, FN, barWidth, 'FaceColor', [1 0.6 0]);      % FN - laranja

set(gca, 'XTick', x, 'XTickLabel', classes);
ylabel('Número de Caixas');
title('Comparação de True Positives, False Positives e False Negatives');
legend('TP', 'FP', 'FN');
grid on;
