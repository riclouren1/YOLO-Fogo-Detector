%% COMPARAÇÃO: YOLOv2 com validação e inputSize 256x256

% 1. Carregar gTruths
g1 = load("gTruth.mat"); g2 = load("gTruth2.mat"); g3 = load("gTruth3.mat");
gTruthFinal = merge(g1.gTruth, g2.gTruth, g3.gTruth);

% 2. Gerar dados anotados
classes = {'ceu_limpo','chama','chama_fumo','fumo','nuvem_neblina'};
dados = objectDetectorTrainingData(gTruthFinal);

% 3. Criar datastores
imds = imageDatastore(dados.imageFilename);
blds = boxLabelDatastore(dados(:, classes));
fullDS = combine(imds, blds);

% 4. Divisão estratificada por classe (70% treino, 30% validação)
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

% 5. Subsets de treino e validação
trainingDS = subset(fullDS, find(trainIdx));
valDS = subset(fullDS, find(valIdx));

% 6. Estimar anchor boxes com treino
inputSize = [256 256 3];
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDS, numAnchors);

% 6.1. Filtrar anchors grandes demais
validIdx = all(anchors <= inputSize(1:2), 2);
anchors = anchors(validIdx, :);
if isempty(anchors)
    error("Nenhum anchor box válido após filtragem para inputSize 256x256.");
end

% 7. Definir arquitetura YOLOv2 com ResNet-18
netBase = resnet18;
featureLayer = 'res5b_relu';
lgraph = yolov2Layers(inputSize, numel(classes), anchors, netBase, featureLayer);

% 8. Opções de treino
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'Plots', 'training-progress', ...
    'ValidationData', valDS, ...
    'ValidationFrequency', 30);

% 9. Treinar o modelo de comparação
detectorComparacao = trainYOLOv2ObjectDetector(trainingDS, lgraph, options);

% 10. Guardar modelo
save('detector_comparacao_yolov2.mat', 'detectorComparacao');
disp(" Modelo YOLOv2 alternativo com validação treinado com sucesso.");

