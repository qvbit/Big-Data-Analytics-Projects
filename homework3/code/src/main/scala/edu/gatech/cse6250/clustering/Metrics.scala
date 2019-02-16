/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.clustering

import org.apache.spark.rdd.RDD
import scala.math.max

object Metrics {
  /**
   * Given input RDD with tuples of assigned cluster id by clustering,
   * and corresponding real class. Calculate the purity of clustering.
   * Purity is defined as
   * \fract{1}{N}\sum_K max_j |w_k \cap c_j|
   * where N is the number of samples, K is number of clusters and j
   * is index of class. w_k denotes the set of samples in k-th cluster
   * and c_j denotes set of samples of class j.
   *
   * @param clusterAssignmentAndLabel RDD in the tuple format
   *                                  (assigned_cluster_id, class)
   * @return
   */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
     * TODO: Remove the placeholder and implement your code here
     */

    val N = clusterAssignmentAndLabel.count().toDouble

    val psum = clusterAssignmentAndLabel.map(x => (x, 1))
      .keyBy(x => x._1)
      .reduceByKey((x, y) => (x._1, x._2 + y._2))
      .map(x => (x._1._1, x._2._2))
      .reduceByKey((x, y) => max(x, y))
      .map(x => x._2)
      .reduce(_ + _)
    psum / N
  }
}
