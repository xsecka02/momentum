package bp.momentum.entity;

import static bp.momentum.BPMomentum.getDoubleArrayFromList;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * A class representing one Adaline neurons layer.
 * 
 * @author pseckarova
 */
public class Layer {
    private final ArrayList<Adaline> neurons;
    private ArrayList<Double> output;
    private ArrayList<Double> input;
    private final float learningRate;
    private final float momentumRate;
    private final float lambda;

    /**
     * A constructor initializing all final fields of this Layer.
     * 
     * @param learningRate A learning rate (mi) parameter given for learning 
     * of this network by user.
     * @param momentumRate A momentum rate (alfa) parameter given for learning 
     * of this network by user.
     * @param lambda A lambda parameter given for this network by user.
     * @param width A count of neurons in this layer.
     * @param inputWidth A count of neurons in previous layer +1 for static 1.0 
     * added to beginning of every input vector.
     */
    public Layer(float learningRate, float momentumRate, float lambda, int width, int inputWidth) {
        this.learningRate = learningRate;
        this.momentumRate = momentumRate;
        this.lambda = lambda;
        this.neurons = new ArrayList<>();
        
        for(int i = width; i>0; i--) {
           neurons.add(new Adaline(inputWidth));
        }
    }
    
    /**
     * Computes output vector of this layer for given input.
     * 
     * @param input A vector of this layer's input values.
     * @return the vector of this layer's neurons' computed outputs.
     */
    public ArrayList<Double> computeOutput(ArrayList<Double> input, PrintWriter log) {
        this.input = input; //stored for computation of neurons' deltas later
        
        output = new ArrayList<>();
        for(Adaline n : neurons) {
            output.add(n.computeOutput(input,lambda));
            log.append(String.format(" %+1.6f ",n.getOutput()));
        }
        return output;
    }
    
    /**
     * Computes error from the difference in this layer's output compared 
     * to the expected. Sets this layer's deltas.
     * 
     * @param expectedOutput A vector of values, that were expected as output 
     * of this layer.
     * @return the computed error of this layer's output.
     */
    public double computeError(ArrayList<Double> expectedOutput, PrintWriter log) {
        double error = 0.0;
        
        int i = 0;
        for(Adaline n : neurons) {
            double diff = expectedOutput.get(i) - n.getOutput();
            n.computeDelta(lambda, diff);
            error += 0.5 * diff * diff;
            log.append("\noutput diff: " + diff);
        }
        
        return error;
    }
    
    /**
     * Computes values for vector of error propagation for the previous neuron layer.
     * One error propagation value for one neuron N from the previous layer is computed
     * as a sum of all weights to the N from all the i neurons M_i of this layer 
     * multiplied by that M_i's delta.
     * 
     * @return the computed error propagation vector.
     */
    public ArrayList<Double> getErrorPropagation() {
        int inputWidth = neurons.get(0).getInputWidth();
        Double[] errPropogationArr = new Double[inputWidth];
        Arrays.fill(errPropogationArr, 0.0);
        
        for (Adaline n : neurons) {
            Double[] weights = n.getInWeights();
            
            for (int i = 0; i<inputWidth; i++) {
                errPropogationArr[i] += n.getDelta()*weights[i];
                i++;
            } 
        }
        
        return new ArrayList(Arrays.asList(errPropogationArr));
    }
    
    /**
     * Computes deltas for all this layer's neurons.
     * 
     * @param errPropagation a vector of error propagation values explained 
     * by the Layer.getErrorPropagation() method. 
     */
    public void computeDeltas(ArrayList<Double> errPropagation) {
        int i = 0;
        for(Adaline n : neurons) {
            n.computeDelta(lambda, errPropagation.get(i++));
        }
    }
    
    /**
     * Computes all input weight changes for all of this layer's neurons.
     */
    public void computeWeightChanges(PrintWriter log) {
        int i = 1;
        for (Adaline n : neurons) {
            log.append("\nneuron " + i++ +": ");
            n.computeWeightChanges(input, learningRate, momentumRate, log);
        }
    }
    
    /**
     * Adjusts all input weights of all of this layer's neurons.
     */
    public void adjustWeights() {
        for (Adaline n : neurons) {
            n.adjustWeights();
        }
    }
    
    public ArrayList<ArrayList<Double>> getWeightChanges() {
        ArrayList<ArrayList<Double>> weightChanges = new ArrayList();
        for(Adaline n : neurons) {
            weightChanges.add(new ArrayList<>(Arrays.asList(n.getLastWeightChange())));
        }
        return weightChanges;
    }
    
    public void setWeightChanges(ArrayList<ArrayList<Double>> weightChanges) {
        int i = 0;
        for(Adaline n : neurons) {
            n.setLastWeightChange(getDoubleArrayFromList(weightChanges.get(i++)));
        }
    }

    public ArrayList<Adaline> getNeurons() {
        return neurons;
    }
}
