package edu.gatech.cse6250.phenotyping

import edu.gatech.cse6250.model.{ Diagnostic, LabResult, Medication }
import org.apache.spark.rdd.RDD

/**
 * @author Hang Su <hangsu@gatech.edu>,
 * @author Sungtae An <stan84@gatech.edu>,
 */
object T2dmPhenotype {

  /** Hard code the criteria */
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  /**
   * Transform given data set to a RDD of patients and corresponding phenotype
   *
   * @param medication medication RDD
   * @param labResult  lab result RDD
   * @param diagnostic diagnostic code RDD
   * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
   */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
     * Remove the place holder and implement your code here.
     * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
     * When testing your code, we expect your function to have no side effect,
     * i.e. do NOT read from file or write file
     *
     * You don't need to follow the example placeholder code below exactly, but do have the same return type.
     *
     * Hint: Consider case sensitivity when doing string comparisons.
     */

    val sc = medication.sparkContext

    val all_patients = (diagnostic.map(x => x.patientID)).union(labResult.map(x => x.patientID)).union(medication.map(x => x.patientID)).distinct()

    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

    /** Find CASE Patients */
    val t1_patients = diagnostic.filter(x => T1DM_DX.contains(x.code)).map(x => x.patientID).distinct()
    val not_t1_patients = all_patients.subtract(t1_patients)

    val t2_patients = diagnostic.filter(x => T2DM_DX.contains(x.code)).map(x => x.patientID).distinct()
    val not_t2_patients = all_patients.subtract(t2_patients)

    val t1_medication = medication.filter(x => T1DM_MED.contains(x.medicine.toLowerCase)).map(x => x.patientID).distinct()
    val not_t1_medication = all_patients.subtract(t1_medication)

    val t2_medication = medication.filter(x => T2DM_MED.contains(x.medicine.toLowerCase)).map(x => x.patientID).distinct()
    val not_t2_medication = all_patients.subtract(t2_medication)

    val path1 = not_t1_patients.intersection(t2_patients).intersection(not_t1_medication)
    val path2 = not_t1_patients.intersection(t2_patients).intersection(t1_medication).intersection(not_t2_medication)

    val path3_a = not_t1_patients.intersection(t2_patients).intersection(t1_medication).intersection(t2_medication)
    val path3_b = medication.map(x => (x.patientID, x)).join(path3_a.map(x => (x, 0)))
    val path3_c = path3_b.map(x => Medication(x._2._1.patientID, x._2._1.date, x._2._1.medicine))
    val path3_t1 = path3_c.filter(x => T1DM_MED.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val path3_t2 = path3_c.filter(x => T2DM_MED.contains(x.medicine.toLowerCase)).map(x => (x.patientID, x.date.getTime())).reduceByKey(Math.min)
    val path3_join = path3_t1.join(path3_t2)
    val path3 = path3_join.filter(x => x._2._1 > x._2._2).map(x => x._1)

    val casePatients_without_label = sc.union(path1, path2, path3)
    val casePatients = casePatients_without_label.map(x => (x, 1))

    /** Find CONTROL Patients */
    val glucose = labResult.filter(x => x.testName.toLowerCase.contains("glucose")).map(_.patientID).distinct()
    val glucoseSet = glucose.collect.toSet
    val haveGlucoseTest = labResult.filter(x => glucoseSet(x.patientID))

    val ab1 = haveGlucoseTest.filter(x => x.testName.equals("hba1c") && x.value >= 6.0).map(x => x.patientID)
    val ab2 = haveGlucoseTest.filter(x => x.testName.equals("hemoglobin a1c") && x.value >= 6.0).map(x => x.patientID)
    val ab3 = haveGlucoseTest.filter(x => x.testName.equals("fasting glucose") && x.value >= 110).map(x => x.patientID)
    val ab4 = haveGlucoseTest.filter(x => x.testName.equals("fasting blood glucose") && x.value >= 110).map(x => x.patientID)
    val ab5 = haveGlucoseTest.filter(x => x.testName.equals("fasting plasma glucose") && x.value >= 110).map(x => x.patientID)
    val ab6 = haveGlucoseTest.filter(x => x.testName.equals("glucose") && x.value > 110).map(x => x.patientID)
    val ab7 = haveGlucoseTest.filter(x => x.testName.equals("glucose, serum") && x.value > 110).map(x => x.patientID)
    val abnormal = sc.union(ab1, ab2, ab3, ab4, ab5, ab6, ab7).distinct()

    val non_abnormal_patients = glucose.subtract(abnormal)

    val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "648", "648",
      "648.01", "648.02", "648.03", "648.04", "791.5", "277.7", "V77.1", "256.4")

    val dm_patients_a = diagnostic.filter(x => DM_RELATED_DX.contains(x.code)).map(x => x.patientID).distinct()
    val dm_patients_b = diagnostic.filter(x => x.code.startsWith("250.")).map(x => x.patientID).distinct()
    val dm_patients = dm_patients_a.union(dm_patients_b)
    val non_dm_patients = all_patients.subtract(dm_patients).distinct()

    val controlPatients_without_label = non_abnormal_patients.intersection(non_dm_patients)
    val controlPatients = controlPatients_without_label.map(x => (x, 2))

    /** Find OTHER Patients */
    val others = all_patients.subtract(casePatients_without_label).subtract(controlPatients_without_label).map(x => (x, 3))

    /** Once you find patients for each group, make them as a single RDD[(String, Int)] */
    val phenotypeLabel = sc.union(casePatients, controlPatients, others)

    /** Return */
    phenotypeLabel
  }
}