%% Avaliação do Modelo Alternativo YOLOv2

% 1. Carregar o modelo alternativo
load('detector_comparacao_yolov2.mat');

% 2. Carregar os gTruths
g1 = load("gTruth.mat"); g2 = load("gTruth2.mat"); g3 = load("gTruth3.mat");
gTruthFinal = merge(g1.gTruth, g2.gTruth, g3.gTruth);

% 3. Gerar tabela com dados anotados
classes = {'ceu_limpo','chama','chama_fumo','fumo','nuvem_neblina'};
dados = objectDetectorTrainingData(gTruthFinal);

% 4. Preparar datastores
imds = imageDatastore(dados.imageFilename);
blds = boxLabelDatastore(dados(:, classes));
fullDS = combine(imds, blds);

% 5. Repetir divisão estratificada (70 treino / 30 validação)
trainIdx = false(height(dados),1);
valIdx = false(height(dados),1);
for i = 1:numel(classes)
    nome = classes{i};
    idxClasse = find(~cellfun(@isempty, dados.(nome)));

    rng(i); idxClasse = idxClasse(randperm(numel(idxClasse)));
    nVal = round(0.3 * numel(idxClasse));
    
    valIdx(idxClasse(1:nVal)) = true;
    trainIdx(idxClasse(nVal+1:end)) = true;
end
valIdx(trainIdx) = false;

% 6. Criar conjunto de validação
valData = dados(valIdx, :);
imdsVal = imageDatastore(valData.imageFilename);
bldsVal = boxLabelDatastore(valData(:, classes));
valDS = combine(imdsVal, bldsVal);

% 7. Mostrar distribuição
fprintf("\nDistribuição de caixas por classe (validação - COMPARAÇÃO):\n");
for i = 1:numel(classes)
    nome = classes{i};
    caixas = valData.(nome);
    total = sum(cellfun(@(x) size(x, 1), caixas));
    fprintf("Classe %-18s : %3d caixas\n", nome, total);
end

% 8. Fazer deteções
results = detect(detectorComparacao, valDS);

% 9. Avaliar métricas
[ap, recall, precision] = evaluateDetectionPrecision(results, valDS);

% 10. Calcular F1-score por classe
fprintf("\n=== mAP e F1-score por Classe (Modelo de Comparação) ===\n");
mapGeral = mean(ap, 'omitnan');
for i = 1:numel(classes)
    r = recall{i}; p = precision{i};
    if isempty(r) || isempty(p)
        f1 = 0;
    else
        f1 = max(2*(p.*r)./(p + r + eps));
    end
    fprintf("Classe %-18s | AP = %.3f | F1-score (máx) = %.3f\n", classes{i}, ap(i), f1);
end
fprintf("mAP geral: %.3f\n", mapGeral);

% 11. Gráfico Precision-Recall
figure;
hold on;
for i = 1:numel(classes)
    r = recall{i}; p = precision{i};
    if ~isempty(r) && ~isempty(p)
        plot(r, p, 'DisplayName', classes{i});
    end
end
xlabel('Recall'); ylabel('Precisão');
title('Curvas Precision-Recall - Modelo de Comparação');
legend('Location', 'best'); grid on;
saveas(gcf, 'comparacao_precision_recall.png');
print(gcf, 'comparacao_precision_recall.pdf', '-dpdf', '-bestfit');

% 12. Exibir exemplos
figure;
for i = 1:6
    subplot(2,3,i);
    I = readimage(imdsVal, i);
    [bboxes, scores, labels] = detect(detectorComparacao, I);
    if ~isempty(bboxes)
        I = insertObjectAnnotation(I, 'rectangle', bboxes, labels);
    end
    imshow(I);
    title(sprintf('Exemplo %d', i));
end
