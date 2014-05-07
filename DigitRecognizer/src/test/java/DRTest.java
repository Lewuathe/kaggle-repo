import junit.framework.TestCase;
import org.junit.Test;

import java.io.*;

/**
 * Created by sasakiumi on 4/26/14.
 */
public class DRTest extends TestCase{

    @Test
    public void testDR() throws IOException {
        double ret = Main.test();
        File file = new File("result.txt");
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(file)));
        pw.println(ret);
        pw.close();
        System.out.printf("Accuracy: %f\n", ret);
        assertTrue(ret > 0.0);
    }
}
