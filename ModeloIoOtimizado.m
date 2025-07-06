%% CNN.m - Treino de detetor YOLOv2 para detetar classes de incêndios 

%% 1. Carregar gTruth
if ~exist('gTruth', 'var')
    g1 = load("gTruth.mat"); gTruth = g1.gTruth;
end
if ~exist('gTruth2', 'var')
    g2 = load("gTruth2.mat"); gTruth2 = g2.gTruth;
end
if ~exist('gTruth3', 'var')
    g3 = load("gTruth3.mat"); gTruth3 = g3.gTruth;
end

% Combinar todos os gTruths
gTruthFinal = merge(gTruth, gTruth2, gTruth3);

%% 2. Gerar dados de treino
dados = objectDetectorTrainingData(gTruthFinal);

%% 3. Preparar datastores
classes = {'ceu_limpo','chama','chama_fumo','fumo','nuvem_neblina'};
imds = imageDatastore(dados.imageFilename);
blds = boxLabelDatastore(dados(:, classes));
fullDS = combine(imds, blds);

%% 4. Divisão aleatória 70% treino, 30% validação
numTotal = height(dados);
rng(42); % Para reprodutibilidade
indices = randperm(numTotal);
nTrain = round(0.7 * numTotal);
trainDS = subset(fullDS, indices(1:nTrain));
valDS   = subset(fullDS, indices(nTrain+1:end));

%% 5. Estimar anchor boxes
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainDS, numAnchors);
inputSize = [416 416 3];
validIdx = all(anchors <= inputSize(1:2), 2);
anchors = anchors(validIdx, :);

if isempty(anchors)
    error("Nenhum anchor box válido gerado.");
end

%% 6. Definir arquitetura YOLOv2
netBase = resnet18;
featureLayer = 'res5b_relu';

lgraph = yolov2Layers(inputSize, numel(classes), anchors, netBase, featureLayer);

%% 7. Opções de treino
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

%% 8. Treinar
detector = trainYOLOv2ObjectDetector(trainDS, lgraph, options);

%% 9. Guardar
save('detector_yolov2_incendios.mat', 'detector');
