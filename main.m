%% Generalized Gradient Descent Script with Regularization and Optional Cross-Validation
clear; close all; clc

capitalize = @(str) regexprep(str, '(\<\w)', '${upper($1)}');

% == PATHS ==
addpath(genpath('core'));
addpath(genpath('utils'));

% == USER CONFIGURATION ==
modelType = "linear";         % Options: "linear", "quadratic"
lossFunction = "lasso";       % Options: "lasso", "ridge"
regularizationType = "lasso"; % Options: "lasso", "ridge", "elastic"
useCrossVal = true;           % true = K-fold, false = single fit

% Set Parameters by Model
switch modelType
    case "linear"
        thetaTrue = [2; -1];
        nOfPoints = 50;
        xLeft = -5; xRight = 5;
        numOutliers = 15;
        maxDiff = 6;

    case "quadratic"
        thetaTrue = [1; -2; 1];
        nOfPoints = 25;
        xLeft = -2; xRight = 2;
        numOutliers = 10;
        maxDiff = 5;

    otherwise
        error('Unsupported model type: %s', modelType);
end

% Set Loss Type Description
switch lossFunction
    case "lasso"
        lossType = "Lasso Loss";
    case "ridge"
        lossType = "Ridge Loss";
    otherwise
        lossType = "Unknown Loss";
end

% Set Regularization
switch regularizationType
    case "lasso"
        alpha = 0.2; beta = 0;
    case "ridge"
        alpha = 0; beta = 0.2;
    case "elastic"
        alpha = 0.2; beta = 0.2;
    otherwise
        error('Unsupported regularization type: %s', regularizationType);
end

% Generate Synthetic Data
noiseVariance = 0.2;
[x, y, t] = CreateGradDescLinRegData(noiseVariance, nOfPoints, xLeft, xRight, thetaTrue);
y = OutliersAddition(y, numOutliers, 'maxDiff', maxDiff);

% == Processing ==
if useCrossVal
    rng(1);
    nFolds = 5;
    sampPerFold = ceil(nOfPoints / nFolds);
    indsData = randperm(nOfPoints);
    validInds = cell(nFolds, 1);
    cvErrs = zeros(nFolds, 1);
    trainErrs = zeros(nFolds, 1);

    idx1 = 1;
    for k = 1:nFolds
        idx2 = min(nOfPoints, idx1 + sampPerFold - 1);
        validInds{k} = idx1:idx2;
        idx1 = idx2 + 1;
    end

    for k = 1:nFolds
        trInds = setdiff(1:nOfPoints, validInds{k});
        trX = x(indsData(trInds)); trY = y(indsData(trInds));
        cvX = x(indsData(validInds{k})); cvY = y(indsData(validInds{k}));
        parT = t(indsData(trInds));

        [trXn, muX, sigmaX] = DataStandardization(trX);
        [trYn, muY, sigmaY] = DataStandardization(trY);

        switch modelType
            case "quadratic"
                switch lossFunction
                    case "lasso"
                        theta = GradDescFitQuad(@LassoLossQuad, trXn, trYn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                        [yh, ~, ~] = LassoLossQuad(theta, trXn);
                        [ypValStd, ~] = LassoLossQuad(theta, (cvX - muX) ./ sigmaX);
                    case "ridge"
                        theta = GradDescFitQuad(@RidgeLossQuad, trXn, trYn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                        [yh, ~, ~] = RidgeLossQuad(theta, trXn);
                        [ypValStd, ~] = RidgeLossQuad(theta, (cvX - muX) ./ sigmaX);
                end
            case "linear"
                switch lossFunction
                    case "lasso"
                        theta = GradDescFitLine(@LassoLossLine, trXn, trYn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                        [yh, ~, ~] = LassoLossLine(theta, trXn, trYn, 'alpha', alpha, 'beta', beta);
                        [ypValStd, ~] = LassoLossLine(theta, (cvX - muX) ./ sigmaX);
                    case "ridge"
                        theta = GradDescFitLine(@RidgeLossLine, trXn, trYn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                        [yh, ~, ~] = RidgeLossLine(theta, trXn, trYn, 'alpha', alpha, 'beta', beta);
                        [ypValStd, ~] = RidgeLossLine(theta, (cvX - muX) ./ sigmaX);
                end
        end

        yp = DataStandardizationInversion(yh, muY, sigmaY);
        ypVal = DataStandardizationInversion(ypValStd, muY, sigmaY);
        yOrig = DataStandardizationInversion(trYn, muY, sigmaY);
        xOrig = DataStandardizationInversion(trXn, muX, sigmaX);

        trainErrs(k) = mean((yOrig - yp).^2);
        cvErrs(k) = mean((cvY - ypVal).^2);

        figure(k)
        hold on; grid on;
        plot(xOrig, parT, 'b')
        plot(xOrig, yOrig, 'r.')
        plot(cvX, cvY, '*k')
        plot(xOrig, yp, '--g')
        xlabel('x'); ylabel('y');
        title(sprintf('%s | %s | %s Reg | Fold %d\nTrain Error = %.4f | Val Error = %.4f', ...
            capitalize(modelType), capitalize(lossType), capitalize(regularizationType), k, trainErrs(k), cvErrs(k)))
        legend('Actual', 'Measured', 'Test', 'Predicted')
    end

else
    [xn, muX, sigmaX] = DataStandardization(x);
    [yn, muY, sigmaY] = DataStandardization(y);

    switch modelType
        case "quadratic"
            switch lossFunction
                case "lasso"
                    theta = GradDescFitQuad(@LassoLossQuad, xn, yn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                    [yh, ~, ~] = LassoLossQuad(theta, xn);
                case "ridge"
                    theta = GradDescFitQuad(@RidgeLossQuad, xn, yn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                    [yh, ~, ~] = RidgeLossQuad(theta, xn);
            end
        case "linear"
            switch lossFunction
                case "lasso"
                    theta = GradDescFitLine(@LassoLossLine, xn, yn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                    [yh, ~, ~] = LassoLossLine(theta, xn, yn, 'alpha', alpha, 'beta', beta);
                case "ridge"
                    theta = GradDescFitLine(@RidgeLossLine, xn, yn, 'verbose', true, 'alpha', alpha, 'beta', beta);
                    [yh, ~, ~] = RidgeLossLine(theta, xn, yn, 'alpha', alpha, 'beta', beta);
            end
    end

    yp = DataStandardizationInversion(yh, muY, sigmaY);
    yOrig = DataStandardizationInversion(yn, muY, sigmaY);
    xOrig = DataStandardizationInversion(xn, muX, sigmaX);

    trainErr = mean((yOrig - yp).^2);

    figure
    hold on; grid on;
    plot(xOrig, t, 'b')
    plot(xOrig, yOrig, 'r.')
    plot(xOrig, yp, '--g')
    xlabel('x'); ylabel('y');
    title(sprintf('%s | %s | %s Reg | Train Error = %.4f', ...
        capitalize(modelType), capitalize(lossType), capitalize(regularizationType), trainErr))
    legend('Actual', 'Measured', 'Predicted')
end
