import com.lewuathe.magi.NeuralNetwork;
import com.lewuathe.magi.Util;
import com.orangesignal.csv.Csv;
import com.orangesignal.csv.CsvConfig;
import com.orangesignal.csv.CsvReader;
import com.orangesignal.csv.CsvWriter;
import com.orangesignal.csv.handlers.ResultSetHandler;
import com.orangesignal.csv.handlers.StringArrayListHandler;

import java.io.*;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.function.BiConsumer;

/**
 * Created by sasakiumi on 4/24/14.
 */
public class Main {

    public static final int TRAIN_NUM = 41000;
    public static final int TEST_NUM = 100;
    public static final int ANS_NUM = 28000;
    public static final int EPOCHS = 15;
    public static final double LEARNING_RATE = 0.1;
    public static final int TRAINING_SET_CYCLE = 1;
    public static final double MIN_VALUE = 0.0000001;

    public static int maxIndex(double[] ds) {
        int maxIndex = 0;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < ds.length; i++) {
            if (max < ds[i]) {
                maxIndex = i;
                max = ds[i];
            }
        }
        return maxIndex;
    }

    public static double test() throws IOException {
        CsvConfig cfg = new CsvConfig();
        cfg.setSkipLines(1);
        CsvReader reader = new CsvReader(new FileReader("train.csv"), cfg);
        List<String> s = reader.readValues();
        double[][] xs = new double[TRAIN_NUM][784];
        double[][] ys = new double[TRAIN_NUM][10];

        int count = 0;
        int[] numLayers = {784, 30, 10};
        NeuralNetwork nn = new NeuralNetwork(numLayers);
        for (int circle = 0; circle < TRAINING_SET_CYCLE; circle++) {
            while (count < TRAIN_NUM) {
                s = reader.readValues();
                for (int i = 1; i <= 784; i++) {
                    xs[count][i - 1] = Double.parseDouble(s.get(i)) / 256.0;
                    //xs[count][i - 1] += MIN_VALUE;
                }
                //xs[count] = Util.standardize(xs[count]);
                for (int i = 0; i < 10; i++) {
                    if (Integer.parseInt(s.get(0)) == i) {
                        ys[count][i] = 1.0;
                    } else {
                        ys[count][i] = 0.0;
                    }
                }
                count++;
            }
        }

        double[][] testxs = new double[TEST_NUM][784];
        double[][] testys = new double[TEST_NUM][10];

        count = 0;
        while (count < TEST_NUM) {
            s = reader.readValues();
            for (int i = 1; i <= 784; i++) {
                testxs[count][i - 1] = Double.parseDouble(s.get(i)) / 256.0;
                //testxs[count][i - 1] += MIN_VALUE;
            }
            //testxs[count] = Util.standardize(testxs[count]);
            for (int i = 0; i < 10; i++) {
                if (Integer.parseInt(s.get(0)) == i) {
                    testys[count][i] = 1.0;
                } else {
                    testys[count][i] = 0.0;
                }
            }
            count++;
        }
        reader.close();
        nn.train(xs, ys, EPOCHS, LEARNING_RATE, 10, testxs, testys, new BiConsumer<double[][], double[][]>() {
            @Override
            public void accept(double[][] doubles, double[][] doubles2) {
                assert doubles.length == doubles2.length;
                int accuracy = 0;
                for (int i = 0; i < doubles.length; i++) {
                    if (maxIndex(doubles[i]) == maxIndex(doubles2[i])) {
                        accuracy++;
                    }
                }
                System.out.printf("Accuracy: %d / %d\n", accuracy, TEST_NUM);
            }
        });

        CsvReader testReader = new CsvReader(new FileReader("test.csv"), cfg);

        count = 0;
        double[][] ansxs = new double[ANS_NUM][784];

        while (count < 28000) {
            s = testReader.readValues();
            for (int i = 0; i < 784 ; i++) {
                ansxs[count][i] = Double.parseDouble(s.get(i)) / 256.0;
            }
            count++;
        }
        testReader.close();

        CsvWriter writer = new CsvWriter(new FileWriter("ans.csv"));
        List<String> header = Arrays.asList("ImageId", "Label");
        writer.writeValues(header);
        for (int i = 0; i < ansxs.length; i++) {
            double[] ret = nn.feedforward(ansxs[i]);
            int ans = maxIndex(ret);
            List<String> line = Arrays.asList(String.valueOf(i + 1), String.valueOf(ans));
            writer.writeValues(line);
        }
        writer.close();



        // Verification
//        int accurate = 0;
//        for (int i = 0; i < TEST_NUM; i++) {
////            for (int j = 0; j < testxs[i].length; j++) {
////                System.out.printf("%f ", testxs[i][j]);
////            }
//            System.out.println("");
//            double[] ans = nn.feedforward(testxs[i]);
//            for (int j = 0; j < ans.length; j++) {
//                System.out.printf("%f ", ans[j]);
//            }
//            System.out.println("");
//            for (int j = 0; j < testys[i].length; j++) {
//                System.out.printf("%f ", testys[i][j]);
//            }
//            System.out.println("\n------------");
//            if (maxIndex(ans) == maxIndex(testys[i])) {
//                accurate++;
//            }
//        }
//        return (double) accurate / TEST_NUM;
        return 1.0;
    }
}
