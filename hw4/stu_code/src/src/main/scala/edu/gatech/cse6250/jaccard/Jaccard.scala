/**
 *
 * students: please put your implementation in this file!
 */
package edu.gatech.cse6250.jaccard

import edu.gatech.cse6250.model._
import edu.gatech.cse6250.model.{ EdgeProperty, VertexProperty }
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /**
     * Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients.
     * Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay. The given patientID should be excluded from the result.
     */

    /** Remove this placeholder and implement your code */
    val patientIds = graph.subgraph(vpred = { case (id, attr) => attr.isInstanceOf[PatientProperty] }).vertices.map(x => x._1).collect().toSet
    val allVertices = graph.collectNeighborIds(EdgeDirection.Out)
    val allOtherPatients = allVertices.filter(v => v._1 != patientID && patientIds.contains(v._1))
    val ourPatientNeighbors = allVertices.filter(v => v._1.toLong == patientID).map(v => v._2).flatMap(a => a).collect().toSet
    val jaccardScore = allOtherPatients.map(v => (v._1, jaccard(ourPatientNeighbors, v._2.toSet)))
    jaccardScore.takeOrdered(10)(Ordering[Double].reverse.on(v => v._2)).map(v => v._1.toLong).toList
  }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
     * Given a patient, med, diag, lab graph, calculate pairwise similarity between all
     * patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where
     * patient-1-id < patient-2-id to avoid duplications
     */

    /** Remove this placeholder and implement your code */
    val sc = graph.edges.sparkContext
    val patientIds = graph.subgraph(vpred = { case (id, attr) => attr.isInstanceOf[PatientProperty] }).vertices.map(x => x._1).collect().toSet
    val patientNeighbors = graph.collectNeighborIds(EdgeDirection.Out).filter(v => patientIds.contains(v._1))
    val cartesianProduct = patientNeighbors.cartesian(patientNeighbors).filter(v => v._1._1 < v._2._1)
    cartesianProduct.map(v => (v._1._1, v._2._1, jaccard(v._1._2.toSet, v._2._2.toSet)))
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /**
     * Helper function
     *
     * Given two sets, compute its Jaccard similarity and return its result.
     * If the union part is zero, then return 0.
     */

    /** Remove this placeholder and implement your code */
    if (a.isEmpty || b.isEmpty) { return 0.0 }
    a.intersect(b).size.toDouble / a.union(b).size.toDouble
  }
}
