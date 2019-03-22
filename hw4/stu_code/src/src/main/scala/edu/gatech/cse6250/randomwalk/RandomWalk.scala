package edu.gatech.cse6250.randomwalk

import edu.gatech.cse6250.model.{ PatientProperty, EdgeProperty, VertexProperty }
import org.apache.spark.graphx._

object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
     * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
     * Return a List of patient IDs ordered by the highest to the lowest similarity.
     * For ties, random order is okay
     */

    val topicSet: VertexId = patientID

    var rankGraph: Graph[Double, Double] = graph
      .outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }
      .mapTriplets(e => 1.0 / e.srcAttr, TripletFields.Src)
      .mapVertices { (id, attr) => if (!(id != topicSet)) 1.0 else 0.0
      }

    def teleport(u: VertexId, v: VertexId): Double = { if (u == v) 1.0 else 0.0 }

    var iteration = 0
    var prevRankGraph: Graph[Double, Double] = null

    while (iteration < numIter) {
      rankGraph.cache()

      val rankUpdates = rankGraph.aggregateMessages[Double](
        ctx => ctx.sendToDst(ctx.srcAttr * ctx.attr), _ + _, TripletFields.Src)

      prevRankGraph = rankGraph
      val rPrb = {
        (src: VertexId, id: VertexId) => alpha * teleport(src, id)
      }

      rankGraph = rankGraph.outerJoinVertices(rankUpdates) {
        (id, oldRank, msgSumOpt) => rPrb(topicSet, id) + (1.0 - alpha) * msgSumOpt.getOrElse(0.0)
      }.cache()

      rankGraph.edges.foreachPartition(x => {})
      prevRankGraph.vertices.unpersist(false)
      prevRankGraph.edges.unpersist(false)

      iteration += 1
    }

    val filteredgraph = graph.subgraph(vpred = (id, attr) => attr.isInstanceOf[PatientProperty]).collectNeighborIds(EdgeDirection.Out).map(x => x._1).collect().toSet
    val top_ten = rankGraph.vertices.filter(x => filteredgraph.contains(x._1)).takeOrdered(11)(Ordering[Double].reverse.on(x => x._2)).map(_._1)
    top_ten.slice(1, top_ten.length).toList
  }
}
