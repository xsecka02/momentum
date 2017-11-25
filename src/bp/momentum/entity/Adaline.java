package bp.momentum.entity;

import java.io.PrintWriter;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


/**
 * A class representing one Adaline neuron.
 *
 * @author pseckarova
 */
public class Adaline {
    private final Double[] inWeights;
    private Double[] lastWeightChange;
    private final Double[] overallWeightChange;
    private final int inputWidth;
    private double delta;
    private double output;

    /**
     * A constructor of one neuron. Initializes all private fields.
     * 
     * @param inputWidth Is width of previous layer +1 for static 1.0 added 
     * to the beginning of every layer input vector.
     */
    public Adaline(int inputWidth) {
        this.inputWidth = inputWidth;
        
        // init input weights with random values
        this.inWeights = new Double[inputWidth];
        initWeights();
        
        this.lastWeightChange = new Double[inputWidth];
        Arrays.fill(lastWeightChange, 0.0);
        
        this.overallWeightChange = new Double[inputWidth];
        Arrays.fill(overallWeightChange, 0.0);
        
        this.output = 0.0;
        this.delta = 0.0;
        
    }
    
    /**
     * Computes and sets this neurons current output. 
     * 
     * @param input
     * @param lambda A lambda parameter given for this network by user.
     * @return the computed output.
     */
    public double computeOutput(ArrayList<Double> input, float lambda) {
        
        double value = 0.0;
        for(int i = 0; i<inputWidth; i++){
            value += input.get(i)*inWeights[i];
        }
        
        output = 1/(1+Math.exp(-lambda*value));
        return output;
    }
    
    /**
     * Computes this neuron's delta for current training.
     * 
     * @param lambda A lambda parameter given for this network by user.
     * @param errProp A propagation of error for error gradient change.
     * @return the computed delta value.
     */
    public double computeDelta(float lambda, double errProp) {
        delta = errProp * lambda * output * (1-output);
        return delta;
    }
    
    /**
     * Computes new weights' changes using this neurons delta. This.delta has 
     * to be computed first. Adjusts overall changes for every input weight,
     * which can be used later for BGD (for SGD would be last weight change enough).
     * 
     * @param input A vector of input values in current run.
     * @param learningRate A learning rate (mi) parameter given for learning 
     * of this network by user.
     * @param momentumRate A momentum (alfa) parameter given for lerning 
     * of this network by user.
     */
    public void computeWeightChanges(ArrayList<Double> input, float learningRate, float momentumRate, PrintWriter log) {
        for(int i = 0; i<inputWidth; i++){
            lastWeightChange[i] = learningRate*delta*input.get(i) + momentumRate*lastWeightChange[i];
            overallWeightChange[i] += lastWeightChange[i];
            log.append(String.format(" %+1.6f (%+1.6f) ",inWeights[i],lastWeightChange[i]));
        }
    }
    
    /**
     * Adjusts this neurons input weights according to overallWeightChange 
     * (has to be computed first). 
     */
    public void adjustWeights() {
        for(int i = 0; i<inputWidth; i++){
            inWeights[i] += overallWeightChange[i];
            overallWeightChange[i] = 0.0;
        }
    }
    
    /**
     * A method setting all the values of this.inWeights to random values from
     * interval -3/sqrt(inputWidth) to 3/sgrt(inputWidth). However the variance 
     * value is set only to 1/sqrt(inputWidth), because Random.nextGausian()
     * method gives about 30% values out of the variance interval. If the random 
     * value gets divided by 3 it is only about 0.3%. 
     */
    private void initWeights() {
        
        Random random = new Random();
        double variance = 1.0/sqrt(inputWidth); 
        
        for(int i = 0; i<inputWidth; i++){
            this.inWeights[i] = (random.nextGaussian()) * variance ;
        }
    }    

    public double getOutput() {
        return output;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public double getDelta() {
        return delta;
    }

    public Double[] getInWeights() {
        return inWeights;
    }

    public Double[] getLastWeightChange() {
        return lastWeightChange;
    }

    public void setLastWeightChange(Double[] lastWeightChange) {
        this.lastWeightChange = lastWeightChange;
    }
    
}
