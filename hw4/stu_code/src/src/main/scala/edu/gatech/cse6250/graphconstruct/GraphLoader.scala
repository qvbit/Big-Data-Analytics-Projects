/**
 * @author Hang Su <hangsu@gatech.edu>.
 */

package edu.gatech.cse6250.graphconstruct

import edu.gatech.cse6250.model._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphLoader {
  /**
   * Generate Bipartite Graph using RDDs
   *
   * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
   * @return: Constructed Graph
   *
   */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
    medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
    val sc = patients.sparkContext

    // Patient vertices
    val vertPatient: RDD[(VertexId, VertexProperty)] = patients
      .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))
    val patient_count = 1000 //for later use in vertex indices

    // Diagnostic vertices
    val most_recent_diag = diagnostics.map(row => ((row.patientID, row.icd9code), row))
      .reduceByKey((row1, row2) => if (row1.date > row2.date) row1 else row2)
      .map(x => x._2)
    val index_diag = most_recent_diag.map(x => x.icd9code).distinct().zipWithIndex.map { case (value, index) => (value, patient_count + index + 1) }
    val vertDiagnostic: RDD[(VertexId, VertexProperty)] = index_diag.map { case (code, index) => (index, DiagnosticProperty(code)) }
    val diagVertId = index_diag.collect.toMap //for later use in edges
    val diag_count = index_diag.count() //for later use in future vertex indices

    //LabResult Vertex
    val most_recent_lab = labResults.map(row => ((row.patientID, row.labName), row))
      .reduceByKey((row1, row2) => if (row1.date > row2.date) row1 else row2)
      .map(row => row._2)
    val index_lab = most_recent_lab.map(x => x.labName).distinct().zipWithIndex.map { case (value, index) => (value, patient_count + diag_count + index + 1) }
    val vertLab: RDD[(VertexId, VertexProperty)] = index_lab.map { case (labname, index) => (index, LabResultProperty(labname)) }
    val labVertId = index_lab.collect.toMap // for later use in edges
    val lab_count = index_lab.count() //for later use in future vertex indices

    //Medication Vertex
    val most_recent_med = medications.map(row => ((row.patientID, row.medicine), row))
      .reduceByKey((row1, row2) => if (row1.date > row2.date) row1 else row2)
      .map(row => row._2)
    val index_med = most_recent_med.map(_.medicine).distinct().zipWithIndex.map { case (value, index) => (value, patient_count + diag_count + lab_count + index + 1) }
    val vertMed: RDD[(VertexId, VertexProperty)] = index_med.map { case (med, index) => (index, MedicationProperty(med)) }
    val medVertId = index_med.collect.toMap // for use in edges

    /**
     * HINT: See Example of Making PatientPatient Edges Below
     *
     * This is just sample edges to give you an example.
     * You can remove this PatientPatient edges and make edges you really need
     */

    // Patient-Diagnostics Edges
    val bcDiagVertId = sc.broadcast(diagVertId)
    val pd_edge_f = most_recent_diag.map(row => Edge(row.patientID.toLong, bcDiagVertId.value(row.icd9code), PatientDiagnosticEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pd_edge_b = most_recent_diag.map(row => Edge(bcDiagVertId.value(row.icd9code), row.patientID.toLong, PatientDiagnosticEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pd_edges = sc.union(pd_edge_f, pd_edge_f)

    //Patient-LabResult Edges
    val bcLabVertId = sc.broadcast(labVertId)
    val pl_edge_f = most_recent_lab.map(row => Edge(row.patientID.toLong, bcLabVertId.value(row.labName), PatientLabEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pl_edge_b = most_recent_lab.map(row => Edge(bcLabVertId.value(row.labName), row.patientID.toLong, PatientLabEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pl_edges = sc.union(pl_edge_f, pl_edge_b)

    //Patient-Medication Edges
    val bcMedVertId = sc.broadcast(medVertId)
    val pm_edge_f = most_recent_med.map(row => Edge(row.patientID.toLong, bcMedVertId.value(row.medicine), PatientMedicationEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pm_edge_b = most_recent_med.map(row => Edge(bcMedVertId.value(row.medicine), row.patientID.toLong, PatientMedicationEdgeProperty(row).asInstanceOf[EdgeProperty]))
    val pm_edges = sc.union(pm_edge_f, pm_edge_b)

    // Making Graph
    val vertices = sc.union(vertPatient, vertDiagnostic, vertLab, vertMed)
    val edges = sc.union(pd_edges, pl_edges, pm_edges)
    val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertices, edges)

    graph
  }
}
