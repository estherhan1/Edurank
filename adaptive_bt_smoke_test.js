/* eslint-disable no-console */
function sigmoid(x){return 1/(1+Math.exp(-x));}
function pairKey(a,b){return [a,b].sort().join('_');}

function fitBradleyTerryScores(videos,matches){
  const n=videos.length;
  const zero={};
  videos.forEach(v=>{zero[v.id]=0;});
  if(n<2||!matches.length)return zero;
  const idx={};
  videos.forEach((v,i)=>{idx[v.id]=i;});
  const wins=Array.from({length:n},()=>Array(n).fill(0));
  const winTotals=Array(n).fill(0);
  matches.forEach(m=>{
    const wi=idx[m.winnerId],li=idx[m.loserId];
    if(wi===undefined||li===undefined||wi===li)return;
    wins[wi][li]+=1;
    winTotals[wi]+=1;
  });
  let theta=Array(n).fill(1);
  const maxIter=200,eps=1e-9;
  for(let it=0;it<maxIter;it++){
    const next=theta.slice();
    for(let i=0;i<n;i++){
      let denom=0;
      for(let j=0;j<n;j++){
        if(i===j)continue;
        const nij=wins[i][j]+wins[j][i];
        if(!nij)continue;
        denom+=nij/Math.max(theta[i]+theta[j],eps);
      }
      if(denom>eps)next[i]=(winTotals[i]+eps)/denom;
    }
    const g=Math.exp(next.reduce((s,v)=>s+Math.log(Math.max(v,eps)),0)/n);
    for(let i=0;i<n;i++)next[i]=Math.max(next[i]/g,eps);
    let delta=0;
    for(let i=0;i<n;i++)delta=Math.max(delta,Math.abs(next[i]-theta[i]));
    theta=next;
    if(delta<1e-7)break;
  }
  const scores={};
  videos.forEach((v,i)=>{scores[v.id]=Math.log(theta[i]);});
  const mean=Object.values(scores).reduce((s,x)=>s+x,0)/Math.max(Object.values(scores).length,1);
  Object.keys(scores).forEach(k=>{scores[k]-=mean;});
  return scores;
}

function buildInitialSparsePairs(videos,videoCounts,pairCounts,batchSize,maxPairRepeats){
  const n=videos.length;
  const ranked=[...videos].sort((a,b)=>(videoCounts[a.id]||0)-(videoCounts[b.id]||0));
  const out=[],seenBatch=new Set();
  for(let i=0;i<n&&out.length<batchSize;i++){
    const a=ranked[i],b=ranked[(i+1)%n];
    const k=pairKey(a.id,b.id);
    if(seenBatch.has(k)||(pairCounts[k]||0)>=maxPairRepeats)continue;
    out.push([a,b]);seenBatch.add(k);
  }
  for(let i=0;i<n&&out.length<batchSize;i++){
    const a=ranked[i],b=ranked[(i+2)%n];
    const k=pairKey(a.id,b.id);
    if(seenBatch.has(k)||(pairCounts[k]||0)>=maxPairRepeats)continue;
    out.push([a,b]);seenBatch.add(k);
  }
  return out;
}

function selectAdaptivePairs(videos,matches,batchSize){
  const minComp=2,maxPairRepeats=3,adjacentBonus=0.35;
  const videoCounts={},pairCounts={};
  videos.forEach(v=>{videoCounts[v.id]=0;});
  matches.forEach(m=>{
    videoCounts[m.winnerId]=(videoCounts[m.winnerId]||0)+1;
    videoCounts[m.loserId]=(videoCounts[m.loserId]||0)+1;
    const k=pairKey(m.winnerId,m.loserId);
    pairCounts[k]=(pairCounts[k]||0)+1;
  });
  if(videos.some(v=>(videoCounts[v.id]||0)<minComp)){
    const starter=buildInitialSparsePairs(videos,videoCounts,pairCounts,batchSize,maxPairRepeats);
    if(starter.length)return starter.slice(0,batchSize);
  }
  const scores=fitBradleyTerryScores(videos,matches);
  const ranking=[...videos].sort((a,b)=>(scores[b.id]||0)-(scores[a.id]||0));
  const rankPos={};
  ranking.forEach((v,i)=>{rankPos[v.id]=i;});
  const candidates=[];
  for(let i=0;i<videos.length;i++)for(let j=i+1;j<videos.length;j++){
    const a=videos[i],b=videos[j],k=pairKey(a.id,b.id);
    const repeats=pairCounts[k]||0;
    if(repeats>=maxPairRepeats)continue;
    const p=sigmoid((scores[a.id]||0)-(scores[b.id]||0));
    const uncertainty=p*(1-p);
    const coverageBonus=1/((videoCounts[a.id]||0)+(videoCounts[b.id]||0)+1);
    const repetitionPenalty=1/(1+repeats);
    const rankGap=Math.abs((rankPos[a.id]||0)-(rankPos[b.id]||0));
    const rankBonus=1+(adjacentBonus/(rankGap+1));
    candidates.push({pair:[a,b],priority:uncertainty*coverageBonus*repetitionPenalty*rankBonus,key:k});
  }
  candidates.sort((x,y)=>y.priority-x.priority);
  return candidates.slice(0,batchSize).map(c=>c.pair);
}

function spearman(xs,ys){
  const n=xs.length;
  let d2=0;
  for(let i=0;i<n;i++){const d=xs[i]-ys[i];d2+=d*d;}
  return 1-(6*d2)/(n*(n*n-1));
}

function run(){
  const n=16,batch=8,rounds=12;
  const videos=Array.from({length:n},(_,i)=>({id:`v${i+1}`}));
  const trueSkills=Array.from({length:n},(_,i)=>2-4*(i/(n-1))); // monotonic true ordering
  const matches=[];
  let seed=42;
  const rand=()=>{
    seed=(1664525*seed+1013904223)>>>0;
    return seed/4294967296;
  };

  for(let r=0;r<rounds;r++){
    const pairs=selectAdaptivePairs(videos,matches,batch);
    pairs.forEach(([a,b])=>{
      const ai=parseInt(a.id.slice(1),10)-1;
      const bi=parseInt(b.id.slice(1),10)-1;
      const p=sigmoid(trueSkills[ai]-trueSkills[bi]);
      const aWins=rand()<p;
      matches.push({winnerId:aWins?a.id:b.id,loserId:aWins?b.id:a.id});
    });
  }

  const est=fitBradleyTerryScores(videos,matches);
  const estRank=[...videos].sort((a,b)=>est[b.id]-est[a.id]).map(v=>parseInt(v.id.slice(1),10)-1);
  const trueRank=Array.from({length:n},(_,i)=>i);
  const estPos=Array(n),truePos=Array(n);
  estRank.forEach((idx,pos)=>{estPos[idx]=pos;});
  trueRank.forEach((idx,pos)=>{truePos[idx]=pos;});
  const rho=spearman(estPos,truePos);
  console.log(`Matches: ${matches.length}`);
  console.log(`Spearman rho vs ground truth: ${rho.toFixed(3)}`);
  if(rho<0.55){
    console.error('Smoke test failed: ranking did not improve enough.');
    process.exit(1);
  }
  console.log('Adaptive BT smoke test passed.');
}

run();
