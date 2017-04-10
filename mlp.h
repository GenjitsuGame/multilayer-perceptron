#ifndef ML1_MLP_H
#define ML1_MLP_H

struct neuron {
    float *m_weights;
    float m_delta;
    int m_nInputs;
    float m_weightedSum;

    neuron();

    ~neuron();

    void create(int t_nInputs);

    float compute(float *t_niputs);
};

struct layer {
    neuron **m_neurons;
    int m_nNeurons;
    int m_nInputs;
    const int m_bias;
    bool m_isOutputLayer;
    float * m_errors;

    layer();

    ~layer();

    void create(int t_nNeurons, int t_nInputs, bool t_isOutputLayer);

    float activate(float);

    float d_activate(float x);

    float *compute(float *t_layerInputs);
};

class mlp {
public:
    mlp();

    ~mlp();

    void create(int *t_modelStruct, int t_nLayers, float t_learningRate, float t_momentum, int t_epochs);

    float *feedForward(float *t_inputs);

    void backPropagate(float *t_outputs, float *t_desiredOutputs);

    void updateWeight();

    float train(float *t_inputs, float *t_desiredOutputs);

private:
    layer *m_inputLayer;
    layer **m_hiddenLayers;
    layer *m_outputLayer;
    int m_nHiddenLayers;
    int m_nLayers;
    int *m_nNeuronsPerHiddenLayers;
    float m_learningRate;
    float m_momentum;
    int m_epochs;
    int m_nOutputs;
    float m_error;
};


#endif //ML1_MLP_H
