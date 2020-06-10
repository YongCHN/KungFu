package session

import (
	"bytes"

	kb "github.com/lsds/KungFu/srcs/go/kungfu/base"
	"github.com/lsds/KungFu/srcs/go/log"
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
)

type strategyList []strategy

type partitionStrategy func(plan.PeerList) strategyList

func (sl strategyList) choose(i int) strategy {
	return sl[i%len(sl)]
}

func (sl strategyList) digestBytes() []byte {
	b := &bytes.Buffer{}
	for _, s := range sl {
		b.Write(s.reduceGraph.DigestBytes())
		b.Write(s.bcastGraph.DigestBytes())
	}
	return b.Bytes()
}

var partitionStrategies = map[kb.Strategy]partitionStrategy{
	kb.Star:                createStarStrategies,
	kb.Clique:              createCliqueStrategies,
	kb.Ring:                createRingStrategies,
	kb.Tree:                createTreeStrategies,
	kb.BinaryTree:          createBinaryTreeStrategies,
	kb.BinaryTreeStar:      createBinaryTreeStarStrategies,
	kb.MultiBinaryTreeStar: createMultiBinaryTreeStarStrategies,
}

func simpleSingleGraphStrategy(bcastGraph *graph.Graph) strategyList {
	return []strategy{
		{
			reduceGraph: plan.GenDefaultReduceGraph(bcastGraph),
			bcastGraph:  bcastGraph,
		},
	}
}

func createStarStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenStarBcastGraph(len(peers), defaultRoot)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createTreeStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenTree(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createBinaryTreeStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenBinaryTree(len(peers))
	return simpleSingleGraphStrategy(bcastGraph)
}

func createBinaryTreeStarStrategies(peers plan.PeerList) strategyList {
	bcastGraph := plan.GenBinaryTreeStar(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createMultiBinaryTreeStarStrategies(peers plan.PeerList) strategyList {
	var sl strategyList
	for _, bcastGraph := range plan.GenMultiBinaryTreeStar(peers) {
		sl = append(sl, strategy{
			reduceGraph: plan.GenDefaultReduceGraph(bcastGraph),
			bcastGraph:  bcastGraph,
		})
	}
	return sl
}

func createCliqueStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		sl = append(sl, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return sl
}

func createRingStrategies(peers plan.PeerList) strategyList {
	k := len(peers)
	var sl strategyList
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		sl = append(sl, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return sl
}

func autoSelect(peers plan.PeerList) kb.Strategy {
	m := make(map[uint32]int)
	for _, p := range peers {
		m[p.IPv4]++
	}
	if len(m) == 1 {
		return kb.Star
	}
	return kb.BinaryTreeStar
}

func genPeerStrategyList(peers plan.PeerList, strategy kb.Strategy) strategyList {
	return partitionStrategies[strategy](peers)
}

func genRootStrategyList(peers plan.PeerList, strategy kb.Strategy) strategyList {
	// masters, parents := peers.PartitionByHost()
	var sl strategyList
	log.Errorf("TODO: genRootStrategyList")
	return sl
}
