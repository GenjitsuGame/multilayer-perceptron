#include <cassert>
#include <cmath>
#include <ctime>
#include "mlp.h"

neuron::neuron() : m_weights(0), m_delta(0), m_weightedSum(0) {

}

neuron::~neuron() {
    delete[] m_weights;
}

void neuron::create(int t_nInputs) {
    assert(t_nInputs);
    srand((unsigned int) time(0));
    m_nInputs = t_nInputs + 1;
    m_weights = new float[m_nInputs];
    m_delta = 0.0f;
    for (int i = 0; i < t_nInputs; ++i) {
        m_weights[i] = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2)));
    }
}


float neuron::compute(float *t_Inputs) {
    float res = 0.0;
    for (int i = 0; i < m_nInputs; ++i) {
        res += t_Inputs[i] * m_weights[i];
    }
    m_weightedSum = res;
    return res;
}

layer::layer() : m_bias(1) {

}

layer::~layer() {
    for (int i = 0; i < m_nNeurons; ++i) {
        delete m_neurons[i];
    }
    delete[] m_neurons;
}

void layer::create(int t_nNeurons, int t_nInputs, bool t_isOutputLayer) {
    m_nNeurons = t_isOutputLayer ? t_nNeurons : t_nNeurons + 1;
    m_nInputs = t_nInputs;
    m_neurons = new neuron *[m_nNeurons];
    m_errors = new float[t_nNeurons];
    for (int i = 0; i < m_nNeurons; ++i) {
        m_neurons[i] = new neuron();
        m_neurons[i]->create(m_nInputs);
    }
}

float layer::activate(float x) {
    return (float) (1.f / (1.f + exp(-x)));
}

float layer::d_activate(float x) {
    return activate(x) * (1.f - activate(x));
}

float *layer::compute(float *t_layerInputs) {
    float *res = new float[m_nNeurons];

    if (m_isOutputLayer) {
        for (int i = 0; i < m_nNeurons; ++i) {
            res[i] = m_neurons[i]->compute(t_layerInputs);
            res[i] = activate(res[i]);
        }
    } else {
        float *inputsWithBias = new float[m_nInputs + 1];

        for (int i = 0; i < m_nInputs; ++i) {
            inputsWithBias[i] = t_layerInputs[i];
        }
        inputsWithBias[m_nInputs] = 1.f;

        for (int i = 0; i < m_nNeurons; ++i) {
            res[i] = m_neurons[i]->compute(inputsWithBias);
            res[i] = activate(res[i]);
        }
    }

    return res;
}

mlp::mlp() : m_hiddenLayers(0) {

}

mlp::~mlp() {

}

void mlp::create(int *t_modelStruct, int t_nLayers, float t_learningRate, float t_momentum, int t_epochs) {
    m_learningRate = t_learningRate;
    m_momentum = t_momentum;
    m_nHiddenLayers = t_nLayers - 2;
    m_nLayers = t_nLayers;
    m_nOutputs = t_modelStruct[m_nLayers - 1];
    m_epochs = t_epochs;
    m_hiddenLayers = new layer *[m_nHiddenLayers];

    for (int i = 1; i < m_nHiddenLayers + 1; ++i) {
        m_hiddenLayers[i - 1] = new layer();
        m_hiddenLayers[i - 1]->create(t_modelStruct[i], t_modelStruct[i - 1], false);
    }
    m_outputLayer = new layer();
    m_outputLayer->create(t_modelStruct[t_nLayers - 1], t_modelStruct[t_nLayers - 2], true);
}

float *mlp::feedForward(float *t_inputs) {
    float *res = m_hiddenLayers[0]->compute(t_inputs);
    for (int i = 1; i < m_nHiddenLayers; ++i) {
        res = m_hiddenLayers[i]->compute(res);
    }
    return m_outputLayer->compute(res);
}

void mlp::backPropagate(float *t_outputs, float* t_desiredOutputs) {
    //float squaredOutputNeuronErrors = 0.f;
    float outputError[m_outputLayer->m_nNeurons];
    int i = 0;
    for (i = 0; i < m_outputLayer->m_nNeurons; ++i) {
        m_outputLayer->m_errors[i] = m_outputLayer->d_activate(m_outputLayer->m_neurons[i]->m_weightedSum) * (t_desiredOutputs[i] - t_outputs[i]);
        //squaredOutputNeuronErrors += (t_desiredOutputs[i] - t_outputs[i]) * (t_desiredOutputs[i] - t_outputs[i]);
    }
    //float outputLayerError = 0.5 * squaredOutputNeuronErrors;

    for (i = m_nHiddenLayers - 1; i >= 0; --i) {
        for (int j = 0; j < m_hiddenLayers[i]->m_nNeurons; ++j) {
            m_hiddenLayers[i]->m_errors[j] = m_hiddenLayers[i]->d_activate(m_hiddenLayers[i]->m_neurons[j]->m_weightedSum);
            float localSum = 0.f;
            if (i == m_nHiddenLayers - 1) {
                for (int k = 0; k < m_outputLayer->m_nNeurons; ++k) {
                    for (int l = 0; l < m_outputLayer->m_neurons[k]->m_nInputs ; ++l) {
                        localSum += m_outputLayer->m_neurons[k]->m_weights[l] * m_outputLayer->m_errors[k];
                    }
                }
            } else {
                for (int k = 0; k < m_hiddenLayers[i + 1]->m_nNeurons; ++k) {
                    for (int l = 0; l < m_hiddenLayers[i + 1]->m_neurons[k]->m_nInputs ; ++l) {
                        localSum += m_hiddenLayers[i + 1]->m_neurons[k]->m_weights[l] * m_hiddenLayers[i + 1]->m_errors[k];
                    }
                }
            }
            m_hiddenLayers[i]->m_errors[j] *= localSum;
        }
    }

}

void mlp::updateWeight() {

}

float mlp::train(float *t_inputs, float *t_desiredOutputs) {
    return 0;
}
