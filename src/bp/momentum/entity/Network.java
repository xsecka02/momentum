package bp.momentum.entity;

import java.io.PrintWriter;
import java.util.ArrayList;

/**
 *
 * @author pseckarova
 */
public class Network {
    private ArrayList<Layer> layers;
    
    public Network(float learning_rate, float momentum_rate, float lambda, ArrayList<Integer> layer_configuration) {
        this.layers = new ArrayList<>();
        
        // store width of network input as first layer's input width
        int prev = layer_configuration.get(0); 
        layer_configuration.remove(0);
        
        for(Integer l : layer_configuration){
            layers.add(new Layer(learning_rate, momentum_rate,lambda,l,prev+1));
            prev = l;
        }
    }
    
    public double train(ArrayList<Double> input, ArrayList<Double> expectedOutput, PrintWriter log) {
        
        run(input, log);
                
        double error = propagateErrorInDeltas(expectedOutput, log);
        int i = 1;
        log.append("\n\n== WEIGHTS ==");
        for (Layer l : layers) {
            log.append("\n= Layer " + i++ +" =");
            l.computeWeightChanges(log);
        }
        
        return error;
    }
    
    public void run(ArrayList<Double> input, PrintWriter log) {
        ArrayList<Double> currentValues = new ArrayList<>(input);
        
        int i = 1;
        log.append("\n== OUTPUTS ==");
        // get current network response to given input vector
        for(Layer l : layers) {
            // add static 1 to the beginning of every input vector
            currentValues.add(1.0); 
            
            // store output of the current layer, input = output of the previous one
            log.append("\nLayer " + i++ +":");
            currentValues = l.computeOutput(currentValues, log);
        }
    }
    
    private double propagateErrorInDeltas(ArrayList<Double> expectedOutput, PrintWriter log) {
        Layer lastLayer = layers.get(layers.size()-1);
        double error = lastLayer.computeError(expectedOutput, log);
        ArrayList<Double> errPropagation = lastLayer.getErrorPropagation();
        
        for (int i = layers.size()-2; i>=0;i--) {
            Layer currentLayer = layers.get(i);
            currentLayer.computeDeltas(errPropagation);
            errPropagation = currentLayer.getErrorPropagation();
        }
        return error;
    }
    
    public void adjustWeights() {
        for (Layer l : layers) {
            l.adjustWeights();
        }
    }    

    public ArrayList<ArrayList<ArrayList<Double>>> getWeightChanges() {
        ArrayList<ArrayList<ArrayList<Double>>> weightChanges = new ArrayList();
        for (Layer l : layers) {
            weightChanges.add(l.getWeightChanges());
        }
        return weightChanges;
    }
    
    public void setWeightChanges(ArrayList<ArrayList<ArrayList<Double>>> weightChanges) {
        int i = 0;
        for (Layer l : layers) {
            l.setWeightChanges(weightChanges.get(i++));
        }
    }
}
