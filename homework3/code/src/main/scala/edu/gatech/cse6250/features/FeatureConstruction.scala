package edu.gatech.cse6250.features

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su
 */
object FeatureConstruction {

  /**
   * ((patient-id, feature-name), feature-value)
   */
  type FeatureTuple = ((String, String), Double)

  /**
   * Aggregate feature tuples from diagnostic with COUNT aggregation,
   *
   * @param diagnostic RDD of diagnostic
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    diagnostic.map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation,
   *
   * @param medication RDD of medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    medication.map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
  }

  /**
   * Aggregate feature tuples from lab result, using AVERAGE aggregation
   *
   * @param labResult RDD of lab result
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val lab_sum = labResult.map(x => ((x.patientID, x.testName), x.value)).reduceByKey(_ + _)
    val lab_count = labResult.map(x => ((x.patientID, x.testName), 1.0)).reduceByKey(_ + _)
    val lab_features = lab_sum.join(lab_count).map(x => (x._1, x._2._1 / x._2._2))
    lab_features
  }

  /**
   * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
   * available in the given set only and drop all others.
   *
   * @param diagnostic   RDD of diagnostics
   * @param candiateCode set of candidate code, filter diagnostics based on this set
   * @return RDD of feature tuples
   */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candiateCode: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val dfilt = diagnostic.filter(x => candiateCode.contains(x.code))
    val dfilt_features = dfilt.map(x => ((x.patientID, x.code), 1.0)).reduceByKey(_ + _)
    dfilt_features
  }

  /**
   * Aggregate feature tuples from medication with COUNT aggregation, use medications from
   * given set only and drop all others.
   *
   * @param medication          RDD of diagnostics
   * @param candidateMedication set of candidate medication
   * @return RDD of feature tuples
   */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val mfilt = medication.filter(x => candidateMedication.contains(x.medicine))
    val mfilt_features = mfilt.map(x => ((x.patientID, x.medicine), 1.0)).reduceByKey(_ + _)
    mfilt_features
  }

  /**
   * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
   * given set of lab test names only and drop all others.
   *
   * @param labResult    RDD of lab result
   * @param candidateLab set of candidate lab test name
   * @return RDD of feature tuples
   */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val labfilt = labResult.filter(x => candidateLab.contains(x.testName))
    val lab_sum = labfilt.map(x => ((x.patientID, x.testName), x.value)).reduceByKey(_ + _)
    val lab_count = labResult.map(x => ((x.patientID, x.testName), 1.0)).reduceByKey(_ + _)
    val lab_features = lab_sum.join(lab_count).map(x => (x._1, x._2._1 / x._2._2))
    lab_features
  }

  /**
   * Given a feature tuples RDD, construct features in vector
   * format for each patient. feature name should be mapped
   * to some index and convert to sparse feature format.
   *
   * @param sc      SparkContext to run
   * @param feature RDD of input feature tuples
   * @return
   */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map */
    val feature_map = feature.map(_._1._2).distinct().collect.zipWithIndex.toMap
    val bfeature_map = sc.broadcast(feature_map)
    /** transform input feature */

    /**
     * Functions maybe helpful:
     * collect
     * groupByKey
     */

    /**
     * TODO implement your own code here and remove existing
     * placeholder code
     */
    val num_features = bfeature_map.value.size
    val bnum_features = sc.broadcast(num_features)
    val grouped = feature.map(x => (x._1._1, (x._1._2, x._2))).groupByKey()

    val result = grouped.map {
      case (patientID, features) =>
        val features_mapped = features.toList.map { case (featureName, featureVal) => (bfeature_map.value(featureName), featureVal) }
        val fVector = Vectors.sparse(bnum_features.value, features_mapped)
        val prepared_row = (patientID, fVector)
        prepared_row
    }
    result
  }
}

