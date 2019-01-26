import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class Frequency {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: Frequency <input path> <output path>");
            System.exit(-1);
        }
        
        // Create a Hadoop job and set the main class
        Job job = Job.getInstance();
        job.setJarByClass(Frequency.class);
        job.setJobName("Frequency");

        // Set the input and output path
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Set the Mapper and Reducer class
        job.setMapperClass(FrequencyMapper.class);
        job.setReducerClass(FrequencyReducer.class);

        // Specify the type of output
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        
        // Run job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}