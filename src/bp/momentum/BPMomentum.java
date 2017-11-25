/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package bp.momentum;

import bp.momentum.entity.Network;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.stream.Stream;

/**
 * A class preparing and guiding the learning demontration.
 * 
 * @author pseckarova
 */
public class BPMomentum {

    public static int OK = 0;
    public static int FAILED = -1;
    
    private static Configuration getConfigFromUser(String[] args) {
        ArrayList<String> argsList = new ArrayList<>(Arrays.asList(args));
        
        Configuration conf = new Configuration();
        try {
            String arg = Stream.of(args)
                .filter(s -> s.contains("-f="))
                .findAny()
                .get();
            argsList.remove(arg);
            if (conf.parseConfigFromFile(arg.substring(3)) != OK) {
                System.err.println("Unable to process the configuration file! "
                        + "Please check it's accesibility and correctness.");
                return null;
            }
            
        } catch (NoSuchElementException e) {
            System.err.println("You didn't set the mandatory parameter -f=configuration_file_name!");
            return null;
        }
        
        conf.modifyFromArgs(argsList);
        
        if(!conf.isTestSetValid()) {
            System.err.println("Invalid test set! Please chcek the input/output "
                    + "vectors in configuration file.");
            return null;
        }
        
        return conf;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        if(args.length < 1 || args[0].contains("-h")) {
            printHelp();
            return;
        }
        
        Configuration conf = getConfigFromUser(args);
        
        if(conf == null) {
            System.out.println("Attempt to set up the network failed.");
            generateConfigFile();
            return;
        }
        
        Network nn = new Network(conf.getLearningRate(), conf.getMomentumRate(), 
                conf.getLambda(), conf.getNetworkTopology());
        
        ArrayList<ArrayList<Double>> inputs = conf.getInputs();
        ArrayList<ArrayList<Double>> outputs = conf.getOutputs();
                
        int i,j = 0;
        double error;
        boolean stepByStep = conf.isStepByStep();
        PrintWriter log = conf.createLogWriter();
        
        do { 
            log.append("\n\n iteration no."+ (++j));
            error = 0.0;
            i = 0;
            for (ArrayList<Double> in : inputs) {
                log.append("\n\n===== INPUT no."+i+" =====\n");
                error += nn.train(in,outputs.get(i++),log);
                nn.adjustWeights();
            }
        } while (error > 0.01);
        log.close();
        //nn.run(inputs.get(0));
    }
    
    public static Double[] getDoubleArrayFromList(List<Double> doubles) {
        Double[] array = new Double[doubles.size()];
        int i = 0;
        for (Double d : doubles) {
            array[i++] = d;
        }
        return array;
    }

    private static void generateConfigFile() {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("Would you like to generate a sample config.txt file? (y/n)");
        try {
            char c = (char) br.read();
            if (c == 'y' || c == 'Y') {
                PrintWriter writer = new PrintWriter("config.txt", "UTF-8");
                writer.println(
                        "lambda\t\t0.5\n"
                        + "learning rate (mi)\t0.7\n"
                        + "momentum rate (alpha)\t0.8\n"
                        + "\n"
                        + "layer widths: (input;first hidden;second hidden;...;output)\n"
                        + "2;3;3;1\n"
                        + "\n"
                        + "inputs:\n"
                        + "1;1\n"
                        + "1;0\n"
                        + "0;1\n"
                        + "0;0\n"
                        + "\n"
                        + "outputs:\n"
                        + "0\n"
                        + "1\n"
                        + "1\n"
                        + "0");
                writer.close();
            }
        } catch (IOException e) {
            System.err.println("Unable to create configuration file. "
                    + "Please check this folder's permissions");
        }
    }

    private static void printHelp() {
        System.out.println(
                          "This program serves for demonstration of Back Propagation neural\n"
                        + "network training, using momentum modification.\n\n"
                        
                        + "Program reads configuration from given file (the values given by arguments \n"
                        + "then override the values from the file), creates a neural network accordingly\n"
                        + "and trains it using BGD and momentum in backpropagation. The log of training\n"
                        + "is written to a log.txt file. Arguments can be given in any order.\n\n"
                        
                        + "The program can be run as:\n"
                        + "   BPMomentum -h\n"
                        + "      or\n"
                        + "   BPMomentum -f=<config_file_name> [-m=<value>]  [-l=<value>] [-a=<value>]\n"
                        + "[-t=<values>] [-s]\n\n"
                        
                        + "where:\n"
                        + "   -h ...prints this help message\n"
                        + "   -s ...step-by-step mode of the training - the log information is written\n"
                        + "         to output and the training pauses after every iteration through\n"
                        + "         the whole training set\n"
                        + "   -m=<value>  ...given double <value> is set as learning rate(mi)\n"
                        + "                  best from interval <0.1,0.9>\n"
                        + "   -a=<value>  ...given double <value> is set as momentum rate(alpha)\n"
                        + "                  best from interval <0.5,0.95>\n"
                        + "   -l=<value>  ...given double <value> is set as lambda for activation function\n"
                        + "   -t=<values> ...network topology configuration set by layer widths, formated\n"
                        + "                  as -t=input;first hidden;second hidden;...;output\n");
    }
}
