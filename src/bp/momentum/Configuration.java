/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package bp.momentum;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author pseckarova
 */
public class Configuration {
    
    public static String NOT_DOUBLE_NUM_PATTERN = "[^0-9.,]";
    
    private float lambda;
    private float learningRate;
    private float momentumRate;
    private boolean stepByStep;
    private ArrayList<Integer> networkTopology;
    private ArrayList<ArrayList<Double>> inputs;
    private ArrayList<ArrayList<Double>> outputs;

    public Configuration() {
        this.lambda = (float)0.5;
        this.learningRate = (float)0.7;
        this.momentumRate = (float)0.7;
        this.stepByStep = false;
        networkTopology = new ArrayList<>();
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
    }
    
    public int parseConfigFromFile(String filename) {
        File file = new File(filename);
        BufferedReader reader = null;
        ParseState state = ParseState.START;

        try {
            reader = new BufferedReader(new FileReader(file));
            String line;

            while ((line = reader.readLine()) != null) {
                switch(state) {
                    case START:
                        if (line.contains("lambda")) {
                            lambda = Float.parseFloat(line.replaceAll("[^0-9.]", ""));
                        } else if (line.contains("learning rate")) {
                            learningRate = Float.parseFloat(line.replaceAll("[^0-9.]", ""));
                        } else if (line.contains("momentum rate")) {
                            momentumRate = Float.parseFloat(line.replaceAll("[^0-9.]", ""));
                        } else if (line.contains("layer widths")) {
                            state = ParseState.TOPOLOGY;
                        }  else if (line.contains("input")) {
                            state = ParseState.INPUTS;
                        } else if (line.contains("output")) {
                            state = ParseState.OUTPUTS;
                        }
                        break;
                        
                    case TOPOLOGY:
                        if (line.contains(";")) {
                            networkTopology = parseIntArraylistFromString(line, ";");
                        } else if (line.contains("input")) {
                            state = ParseState.INPUTS;
                        } else if (line.contains("output")) {
                            state = ParseState.OUTPUTS;
                        }
                        break;
                        
                    case INPUTS:
                        if (line.matches(".*\\d+.*")) {
                            inputs.add(parseDoubleArraylistFromString(line, ";"));
                        } else if (line.contains("layer widths")) {
                            state = ParseState.TOPOLOGY;
                        } else if (line.contains("output")) {
                            state = ParseState.OUTPUTS;
                        }
                        break; 
                        
                    case OUTPUTS:
                        if (line.matches(".*\\d+.*")) {
                            outputs.add(parseDoubleArraylistFromString(line, ";"));
                        } else if (line.contains("layer widths")) {
                            state = ParseState.TOPOLOGY;
                        } else if (line.contains("input")) {
                            state = ParseState.INPUTS;
                        }
                        break;
                }
            }
        } catch (IOException|NumberFormatException e) {
            return BPMomentum.FAILED;
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
                Logger.getLogger(BPMomentum.class.getName()).log(Level.SEVERE, null, e);
            }
        }
        return BPMomentum.OK;
    }
    
    public int modifyFromArgs(ArrayList<String> args) {
        for (String arg : args) {
            try {
                if (arg.contains("-l=")) {
                    lambda = Float.parseFloat(arg.replaceAll(NOT_DOUBLE_NUM_PATTERN, ""));
                } else if (arg.contains("-m=")) {
                    learningRate = Float.parseFloat(arg.replaceAll(NOT_DOUBLE_NUM_PATTERN, ""));
                } else if (arg.contains("-a=")) {
                    momentumRate = Float.parseFloat(arg.replaceAll(NOT_DOUBLE_NUM_PATTERN, ""));
                } else if (arg.contains("-t=")) {
                    arg = arg.substring(3);
                    networkTopology = parseIntArraylistFromString(arg, ";");
                } else if (arg.contains("-s")) {
                    stepByStep = true;
                }
            } catch (NumberFormatException e) {
                return BPMomentum.FAILED;
            }
        }
        return BPMomentum.OK;
    }
    
    public PrintWriter createLogWriter() {
        PrintWriter log;
        if(stepByStep) {
            log = new PrintWriter(System.out);
        } else {
            try {
            log = new PrintWriter("log.txt", "UTF-8");
            } catch (IOException e) {
                log = new PrintWriter(System.out);
            }
        }
        return log;
    }

    private ArrayList<Integer> parseIntArraylistFromString(String s, String separator) {
        ArrayList<Integer> result = new ArrayList<>();
        
        String[] values = s.split(separator);
        for(int i = values.length-1; i>=0; i--) {
            result.add(0,new Integer(values[i]));
        }
        return result;
    }
    
    
    private ArrayList<Double> parseDoubleArraylistFromString(String s, String separator) {
        ArrayList<Double> result = new ArrayList<>();
        if(s.contains(separator)) {
            String[] values = s.split(separator);
            for(int i = values.length-1; i>=0; i--) {
                result.add(0,new Double(values[i]));
            }
        } else {
            result.add(0,new Double(s));
        }
        
        return result;
    }    
    
    public boolean isTestSetValid() {
        if (inputs.size() < 1 || inputs.size() != outputs.size())
            return false;
        
        int inWidth = networkTopology.get(0);
        int outWidth = networkTopology.get(networkTopology.size()-1);
        int i = 0;
        for (ArrayList<Double> in : inputs) {
            if(in.size() != inWidth || outputs.get(i++).size() != outWidth)
                return false;
        }
        
        return true;
    }


    public ArrayList<Integer> getNetworkTopology() {
        return networkTopology;
    }

    public void setNetworkTopology(ArrayList<Integer> networkTopology) {
        this.networkTopology = networkTopology;
    }

    public ArrayList<ArrayList<Double>> getInputs() {
        return inputs;
    }

    public void setInputs(ArrayList<ArrayList<Double>> inputs) {
        this.inputs = inputs;
    }

    public ArrayList<ArrayList<Double>> getOutputs() {
        return outputs;
    }

    public void setOutputs(ArrayList<ArrayList<Double>> outputs) {
        this.outputs = outputs;
    }

    public float getLambda() {
        return lambda;
    }

    public void setLambda(float lambda) {
        this.lambda = lambda;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getMomentumRate() {
        return momentumRate;
    }

    public void setMomentumRate(float momentumRate) {
        this.momentumRate = momentumRate;
    }

    public boolean isStepByStep() {
        return stepByStep;
    }

    public void setStepByStep(boolean verbose) {
        this.stepByStep = verbose;
    }            
}
