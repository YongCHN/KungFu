package session

import (
	"github.com/lsds/KungFu/srcs/go/plan"
	"github.com/lsds/KungFu/srcs/go/plan/graph"
	"github.com/lsds/KungFu/srcs/go/utils/assert"
)

func (sess *Session) SetPeerStrategy(sl strategyList) error {
	sess.Lock()
	defer sess.Unlock()
	assert.OK(sess.barrier())

	ok, err := sess.BytesConsensus(sl.digestBytes(), "kungfu::SetStrategy")
	assert.True(ok)
	assert.OK(err)
	sess.peerStrategies = sl

	assert.OK(sess.barrier())
	return nil
}

func (sess *Session) SimpleSetPeerStrategy(forest []int32) error {
	assert.True(len(forest) == len(sess.peers))
	bg, m, ok := graph.NewGraphFromForestArray(forest)
	assert.True(m == 1)
	assert.True(ok)
	rg := plan.GenDefaultReduceGraph(bg)
	s0 := strategy{reduceGraph: rg, bcastGraph: bg}
	return sess.SetPeerStrategy([]strategy{s0})
}
