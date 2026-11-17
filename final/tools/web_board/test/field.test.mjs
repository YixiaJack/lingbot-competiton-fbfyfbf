import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../public/field/index.html', import.meta.url), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page(data, storage = new Map(), mode = 'coverage') {
  const elements = new Map();
  const listeners = new Map();
  const intervals = [];
  const element = id => {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, { innerHTML: '', textContent: '', style: {},
        classList: { add: x => classes.add(x), remove: x => classes.delete(x), contains: x => classes.has(x) } });
    }
    return elements.get(id);
  };
  const context = vm.createContext({
    console, Date, Set, Map, Intl, encodeURIComponent,
    document: { getElementById: element, addEventListener: (k, v) => {
      const previous=listeners.get(k);listeners.set(k,event=>{if(previous)previous(event);v(event)});
    }, visibilityState: 'hidden' },
    window: { addEventListener() {} }, location: { hash: '' },
    localStorage: { getItem: key => storage.get(key) ?? null, setItem(key,value) { storage.set(key,value); } },
    fetch: () => new Promise(() => {}), setInterval(fn, ms) { intervals.push([fn,ms]); },
  });
  vm.runInContext(script, context);
  if(mode)context.S.planModes={1:mode,2:mode};
  context.S.data = context.prep(structuredClone(data));
  context.S.state = { selftest: {}, evals: {}, checklist: {} };
  return { context, element, listeners, intervals };
}

const record = (n, task, origin, overrides = {}) => ({
  n, ep: n, task, origin, data_kind: 'real', record_id: `T${task}_train${n}`,
  instruction: task === 1 ? 'Put the red apple on the blue plate.' : task === 2 ? 'Store the lipstick in the top compartment.' : 'Brush paper balls into the dustpan.',
  target: task === 1 ? 'red_apple' : task === 2 ? 'lipstick' : null,
  dest: task === 1 ? 'blue' : task === 2 ? 'top' : 'basket',
  source: 'demo', collector: origin === 'public' ? '官方公共真机' : '现场采集',
  len: 300, dur_s: 10, seen: null, time_from: 'unavailable',
  annotation_score: 5, flags: [], dets: [], dets_end: [], target_box: null, dest_box: null,
  visual_status: 'not_evaluated', ds: 'test-dataset', ...overrides,
});

test('automatic refresh retains the last good page after an error and recovers', async () => {
  const {context:c,element,listeners,intervals}=page(fixture);
  c.render();const previous=element('main').innerHTML;
  c.dataLoading=false;
  c.fetch=()=>Promise.reject(new Error('network unavailable'));
  await c.load();
  assert.equal(element('main').innerHTML,previous);
  assert.ok(element('src').textContent.includes('保留'));
  assert.equal(c.dataLoading,false);
  c.fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(structuredClone(fixture))});
  await c.load();
  assert.equal(element('main').innerHTML,previous);
  assert.ok(!element('src').textContent.includes('保留'));
  const updated=structuredClone(fixture);updated.generated='2026-11-14T05:00:00Z';
  updated.episodes.push(record(104,1,'private',{annotation_result:'成功'}));
  c.fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(updated)});
  await c.load();
  assert.equal(c.S.data.episodes.length,5);
  assert.ok(element('main').innerHTML.includes('5 条'));
  assert.ok(intervals.some(([,ms])=>ms===30000));
  assert.ok(listeners.has('visibilitychange'));
});

test('good-only live bundle drives the same counts, positions and recommendations', {skip:!process.env.FIELD_GOOD}, () => {
  const data=JSON.parse(fs.readFileSync(process.env.FIELD_GOOD,'utf8'));
  const own=data.episodes.filter(e=>e.origin==='private');
  assert.equal(data.inventory.private_episodes,own.length);
  assert.equal(new Set(data.episodes.map(e=>e.n)).size,data.episodes.length);
  assert.equal(new Set(own.map(e=>e.trajectory_sha256)).size,own.length);
  assert.ok(own.every(e=>data.quality_policy.good_dataset_ids.includes(e.ds)));
  assert.ok(own.every(e=>data.quality_policy.accepted_results.includes(e.annotation_result)));
  assert.ok(own.every(e=>!data.quality_policy.excluded_quality.includes(e.annotation_quality)));
  assert.equal(data.inventory.total_frames,data.episodes.reduce((n,e)=>n+e.len,0));
  const {context:c}=page(data);
  assert.ok(c.vReal().includes('自动同步'));
  assert.ok(c.vReal().includes('筛选结果'));
  for(const t of [1,2,3]){
    c.S.task=t;c.S.posOrigin='private';
    assert.equal(c.spatialRows().length,own.filter(e=>e.task===t).length);
    assert.equal(c.byTask(t).length,data.episodes.filter(e=>e.task===t).length);
    assert.ok(c.collectionPlan(t).prompt.length>20);
  }
});
const fixture = {
  generated: '2026-11-14T04:00:00Z', title: '真机测试', image_size: [320, 180],
  detector: { status: 'not_evaluated' }, backgrounds: {},
  episodes: [record(100, 1, 'public'), record(101, 2, 'public'), record(102, 3, 'public'),
    record(103, 1, 'private', { instruction: 'Place the lemon on the pink plate.', target: 'lemon', dest: 'pink', annotation_quality: '差', annotation_score: null })],
};

test('real-data default shows task totals, search, source and original annotations', () => {
  const { context: c } = page(fixture);
  assert.equal(c.S.tab, 'real');
  const rendered = c.vReal();
  for (const expected of ['任务 1', '任务 2', '任务 3', '原始提示词', '原始标注', '官方公共数据', '现场自采', '评分 5', '差'])
    assert.ok(rendered.includes(expected), expected);
  assert.equal(c.realRows().length, 2);
  c.S.origin = 'private';
  assert.equal(c.realRows().length, 1);
  c.S.query = 'lemon';
  assert.equal(c.realRows()[0].record_id, 'T1_train103');
  c.S.query = 'not-present';
  assert.equal(c.realRows().length, 0);
});

test('task filter and pagination do not leak other tasks or invalid pages', () => {
  const data = { ...fixture, episodes: Array.from({ length: 61 }, (_, i) => record(i, 2, 'public')) };
  const { context: c } = page(data);
  c.S.task = 2;
  c.S.page = 2;
  assert.equal((c.vReal().match(/class="click" data-ep/g) || []).length, 1);
  c.S.page = 99;
  c.vReal();
  assert.equal(c.S.page, 2);
  c.S.task = 1;
  assert.equal(c.realRows().length, 0);
});

test('undated public imports do not count as current-shift collection', () => {
  const { context: c } = page({ ...fixture, episodes: fixture.episodes.slice(0, 3) });
  assert.equal(c.lastShift().all.length, 0);
  assert.equal(c.epT(c.S.data.episodes[0]), null);
  assert.ok(c.vShift().includes('采集时间未提供'));
  assert.ok(!c.vShift().includes('上一班每条录制比之前慢'));
});

test('unrun visual checks are explicitly pending, not detector failures', () => {
  const { context: c } = page(fixture);
  const rendered = c.vCoverage();
  assert.ok(rendered.includes('空间覆盖待核验'));
  assert.ok(!rendered.includes('格偏少'));
  assert.ok(c.vQuality().includes('没有提示不代表质量已通过'));
  assert.ok(c.S.data.episodes.every(e => !e.q.includes('target_not_detected')));
});

test('drawer looks up stable IDs, preserves raw prompt, and versions thumbnail URLs', () => {
  const { context: c, element } = page(fixture);
  c.openEp(101);
  assert.ok(element('dcard').innerHTML.includes('T2_train101'));
  assert.ok(element('dcard').innerHTML.includes('Store the lipstick'));
  assert.ok(element('dcard').innerHTML.includes('thumbs%2F101_end.jpg'));
  assert.ok(element('dcard').innerHTML.includes('&amp;v='));
});

test('episode instructions and annotations are HTML-escaped', () => {
  const data = { ...fixture, episodes: [record(0, 1, 'public', { instruction: '<img src=x onerror=alert(1)>', annotation_quality: '<script>bad</script>' })] };
  const { context: c } = page(data);
  const rendered = c.vReal();
  assert.ok(rendered.includes('&lt;img'));
  assert.ok(rendered.includes('&lt;script&gt;'));
  assert.ok(!rendered.includes('<script>bad'));
});

test('actual exported dataset matches expected task counts and all data views render', { skip: !process.env.FIELD_DATA }, () => {
  const data = JSON.parse(fs.readFileSync(process.env.FIELD_DATA, 'utf8'));
  assert.equal(data.episodes.length, 647);
  assert.deepEqual([1, 2, 3].map(t => data.episodes.filter(e => e.task === t).length), [215, 216, 216]);
  assert.equal(data.episodes.reduce((sum, e) => sum + e.len, 0), 405703);
  assert.ok(data.episodes.every(e => e.has_first_frame && e.has_last_frame));
  assert.ok(data.episodes.every(e => e.task === 3 || (e.target && e.dest)));
  assert.deepEqual(data.prompt_policy.tasks['1'].templates.map(t=>t.weight),[41,43,46,43,42]);
  assert.deepEqual(data.prompt_policy.tasks['2'].templates.map(t=>t.weight),[48,42,43,40,43]);
  const { context: c, element } = page(data);
  for (const tab of ['real', 'shift', 'overview', 'coverage', 'analysis', 'quality']) {
    c.S.tab = tab;
    for (const task of [1, 2, 3]) {
      c.S.task = task;
      c.render();
      assert.ok(element('main').innerHTML.length > 100, `${tab}/${task}`);
    }
  }
  for (const task of [1,2,3]) {
    c.S.task=task;
    const plan=c.collectionPlan(task);
    assert.ok(plan.prompt.length>20);
    assert.ok(!plan.prompt.includes('{object}'));
    assert.equal(c.collectionPlan(task),plan,'rerender should preserve selected prompt');
    if(task<3){
      const spec=data.prompt_policy.tasks[String(task)];
      assert.ok(plan.prompt.includes(c.planningPhrases(task,spec)[plan.object]));
      assert.ok(plan.prompt.includes(plan.destination));
    } else assert.ok(data.prompt_policy.tasks['3'].templates.some(t=>t.text===plan.prompt));
    assert.ok(c.vPlan().includes('复制 prompt'));
  }
  const task2=data.episodes.filter(e=>e.task===2);
  const counts=c.targetCounts(2,task2);
  assert.equal(Object.keys(counts).length,10);
  assert.equal(counts.brush_small,0);
  assert.equal(counts.glitter_compact,0);
  assert.equal(counts.brush_large,23);
  assert.equal(Object.values(counts).reduce((a,b)=>a+b,0),216);
  c.S.task=2;
  assert.ok(c.vCoverage().includes('小化妆刷'));
  assert.ok(c.vCoverage().includes('亮片盒'));
  assert.ok(c.vCoverage().includes('不代表画面里没有它'));
  c.S.planHistory={version:1,tasks:{}};
  const suggested=new Set();
  for(let i=0;i<30;i++){
    delete c.S.plans[2];
    const plan=c.collectionPlan(2);
    if(i<6){
      assert.ok(['brush_small','glitter_compact'].includes(plan.object));
      assert.ok(plan.reason.includes('取自赛题物品清单'));
    }
    suggested.add(plan.object+'|'+plan.destination);
    assert.ok(!plan.prompt.includes('undefined'));
  }
  assert.equal(suggested.size,30,'all cosmetic/level combinations rotate before repeating');
});

test('weighted sampling honors each official probability interval',()=>{
  const {context:c}=page(fixture);
  const templates=[{id:'a',weight:41},{id:'b',weight:43},{id:'c',weight:46},{id:'d',weight:43},{id:'e',weight:42}];
  const counts={};
  for(let i=0;i<215;i++){
    const id=c.weightedTemplate(templates,(i+.5)/215).id;
    counts[id]=(counts[id]||0)+1;
  }
  assert.deepEqual(Object.values(counts),templates.map(t=>t.weight));
});

test('next collection prioritizes missing own combinations without claiming collection',()=>{
  const policy={tasks:{'1':{reference_episodes:1,templates:[{id:'put',text:'Put the {object} on the {destination} plate.',weight:1}],object_phrases:{red_apple:'red apple',lemon:'lemon'},destinations:['blue']}}};
  const data={...fixture,prompt_policy:policy,episodes:[record(1,1,'private'),record(2,1,'public')]};
  const {context:c}=page(data);
  const plan=c.collectionPlan(1);
  assert.equal(plan.object,'lemon');
  assert.equal(plan.destination,'blue');
  assert.equal(plan.prompt,'Put the lemon on the blue plate.');
  assert.equal(c.S.data.episodes.length,2);
  assert.ok(c.vPlan().includes('差或未标注'));
});

const task2Policy={reference_episodes:216,templates:[
  {id:'put',text:'Put the {object} in the {destination} compartment.',weight:48},
  {id:'place',text:'Place the {object} inside the {destination} level of the organizer.',weight:42},
  {id:'store',text:'Store the {object} in the {destination} compartment.',weight:43},
  {id:'take',text:'Take the {object} and put it in the {destination} section.',weight:40},
  {id:'organize',text:'Organize the {object} into the {destination} compartment.',weight:43}],
  object_phrases:{lotion_bottle:'beige lotion pump bottle',lip_gloss:'beige lip gloss tube',lipstick:'black lipstick with a clear cap',
    eyeshadow_palette:'rectangular multicolor eyeshadow palette',brush_large:'large black and gold makeup brush',
    highlighter:'light-colored round highlighter compact',contour:'dark-colored round contour compact',powder_puff:'pink makeup puff'},
  destinations:['top','middle','bottom']};
const planningFixture={...fixture,prompt_policy:{tasks:{
  '1':{reference_episodes:2,templates:[{id:'put',text:'Put the {object} on the {destination} plate.',weight:1},
    {id:'place',text:'Place the {object} onto the {destination} plate.',weight:1}],object_phrases:{red_apple:'red apple',lemon:'lemon'},destinations:['blue','white']},
  '2':task2Policy,
  '3':{reference_episodes:4,templates:[{id:'a',text:'Brush all paper balls into the dustpan, then pour them into the basket. Return both arms home.',weight:2},
    {id:'b',text:'Sweep the paper balls into the dustpan and empty it into the basket. Return both arms home.',weight:1},
    {id:'c',text:'Clear the table using the brush and dustpan; empty the balls into the basket. Return both arms home.',weight:1}]}
}}};
function clickNext(p,task){
  p.context.S.task=task;
  p.listeners.get('click')({target:{closest:selector=>selector==='[data-next-plan]'?{}:null}});
  return p.context.collectionPlan(task);
}

function evaluationPage(storage=new Map(),mode=null){
  const p=page(planningFixture,storage,mode),c=p.context;
  c.S.data.prompt_policy.tasks['1'].object_phrases=Object.fromEntries(Array.from(c.T1_OBJ,o=>[o,o.replaceAll('_',' ')]));
  delete c.S.data.prompt_policy.tasks['1'].object_phrases.starfruit;
  delete c.S.data.prompt_policy.tasks['1'].object_phrases.green_chili;
  c.S.data.prompt_policy.tasks['1'].destinations=['white','yellow','pink','purple','blue'];
  return p;
}

test('evaluation mode defaults on and rotates all 20 observed scenes with consistent prompt/setup',()=>{
  const p=evaluationPage(),c=p.context,initialCount=c.S.data.episodes.length;
  for(const task of [1,2]){
    assert.equal(c.planMode(task),'evaluation');const ids=[];let previous=null;
    for(let i=0;i<10;i++){
      const plan=clickNext(p,task);ids.push(plan.scene.id);
      assert.ok(plan.materials.includes(plan.object));
      assert.ok(plan.prompt.includes(plan.destination));
      assert.ok(!/undefined|\{object\}|\{destination\}/.test(plan.prompt));
      assert.equal(plan.mode,'evaluation');
      assert.equal(c.collectionPlan(task),plan,'render/copy must preserve this exact setup');
      if(previous)assert.notEqual(plan.prompt,previous.prompt);
      if(task===1){
        assert.equal(plan.plateColors.length,plan.plateCount);
        assert.equal(new Set(plan.plateColors).size,plan.plateCount);
        assert.ok(plan.plateColors.includes(plan.destination));
        if(plan.scene.plates)assert.equal(plan.plateCount,plan.scene.plates);
        if(plan.materials.some(o=>['cucumber','red_chili','green_chili','eggplant','banana','carrot'].includes(o)))assert.ok(plan.setup.includes('横放'));
      }else{
        assert.equal(plan.organizerPosition,plan.scene.organizer);
        assert.ok(plan.setup.includes('桌面'+plan.scene.organizer));
        assert.ok(plan.setup.includes(c.LEVEL_ZH[plan.destination]));
        assert.ok(!/left|right|corner/.test(plan.prompt),'table position must not replace the target compartment');
      }
      previous=plan;
    }
    assert.deepEqual(ids,[1,2,3,4,5,6,7,8,9,10]);
    assert.equal(clickNext(p,task).scene.id,1);
  }
  assert.equal(c.S.data.episodes.length,initialCount,'observed scenarios are not fabricated training episodes');
  assert.equal(c.planMode(3),'coverage');
});

test('same-color, two-column and unspecified observation details are retained explicitly',()=>{
  const p=evaluationPage(),c=p.context;
  c.setEvalScene(1,'9');let plan=c.collectionPlan(1);
  assert.deepEqual(Array.from(plan.materials),['carrot','pumpkin','orange','persimmon']);assert.ok(plan.setup.includes('橙色外观'));
  c.setEvalScene(1,'10');plan=c.collectionPlan(1);
  assert.ok(plan.materials.includes('red_apple'));assert.ok(plan.setup.includes('红色外观'));assert.equal(plan.plateCount,2);
  c.setEvalScene(1,'8');plan=c.collectionPlan(1);
  assert.equal(plan.plateCount,2);assert.equal(clickNext(p,1).plateCount,3);assert.ok(plan.scene.note.includes('未写盘数'));
  c.setEvalScene(1,'2');plan=c.collectionPlan(1);assert.ok(plan.materials.includes('red_chili'));
  assert.ok(clickNext(p,1).materials.includes('green_chili'));
  c.setEvalScene(1,'3');plan=c.collectionPlan(1);assert.ok(plan.scene.raw.includes('百花果'));assert.ok(plan.scene.note.includes('暂按百香果'));
  c.setEvalScene(2,'7');plan=c.collectionPlan(2);
  assert.ok(plan.setup.includes('最左边和最右边'));assert.equal(plan.targetSide,'左');assert.equal(clickNext(p,2).targetSide,'右');
  c.setEvalScene(2,'10');plan=c.collectionPlan(2);assert.equal(plan.materials.length,2);assert.ok(plan.scene.note.includes('同一件'));
});

test('manual scene generation keeps official weights and all target levels independent of organizer position',()=>{
  const p=evaluationPage(),c=p.context;c.setEvalScene(2,'1');
  c.S.data.episodes=['lotion_bottle','lipstick','lip_gloss'].flatMap((o,i)=>['top','middle'].map((d,j)=>record(i*2+j,2,'private',{target:o,dest:d})));
  let plan=c.collectionPlan(2);assert.equal(plan.organizerPosition,'左上角');assert.equal(plan.destination,'bottom');
  const weighted=c.weightedTemplate;let offset=0;
  for(const expected of task2Policy.templates){
    c.weightedTemplate=templates=>{
      assert.deepEqual(Array.from(templates,t=>t.weight),[48,42,43,40,43]);
      return weighted(templates,(offset+expected.weight/2)/216);
    };
    plan=clickNext(p,2);assert.equal(plan.scene.id,1);assert.equal(plan.template,expected.id);
    assert.equal(plan.probability,expected.weight/216);offset+=expected.weight;
    assert.equal(plan.prompt,expected.text.replace('{object}',task2Policy.object_phrases[plan.object]).replace('{destination}',plan.destination));
  }
  const text=c.collectionSetupText(plan);assert.ok(text.includes(plan.prompt));assert.ok(text.includes('桌面左上角'));
  c.S.task=2;const view=c.vPlan();assert.ok(view.includes('复制采集安排'));assert.ok(view.includes('原始记录'));
});

test('scene selection, mode and rotation survive reload while fresh data reprioritizes scene targets',()=>{
  const storage=new Map(),p=evaluationPage(storage),c=p.context;
  c.setEvalScene(2,'7');c.collectionPlan(2);
  const next=evaluationPage(storage).context;
  assert.equal(next.evalSceneFilter(2),'7');assert.equal(next.collectionPlan(2).targetSide,'右');
  next.setPlanMode(2,'coverage');assert.equal(next.collectionPlan(2).mode,'coverage');
  assert.equal(evaluationPage(storage).context.planMode(2),'coverage');
  c.setEvalScene(2,'auto');c.setPlanMode(2,'evaluation');const first=c.collectionPlan(2);
  c.S.data.episodes.push(record(2000,2,'private',{target:first.object,dest:first.destination}));
  const refreshed=c.collectionPlan(2);assert.notEqual(refreshed.revision,first.revision);
  assert.notEqual(refreshed.scene.id,first.scene.id,'new data should not restart all scene visits at group 1');
  assert.equal(refreshed.basis.private_episodes,1);
});

test('actual latest data renders every evaluation scene including small brushes, starfruit and green chili',{skip:!process.env.FIELD_GOOD},()=>{
  const data=JSON.parse(fs.readFileSync(process.env.FIELD_GOOD,'utf8')),p=page(data,new Map(),null),c=p.context;
  for(const task of [1,2])for(let id=1;id<=10;id++){
    c.S.task=task;c.setEvalScene(task,String(id));const plan=c.collectionPlan(task);
    assert.ok(plan&&plan.prompt.length>20);assert.ok(plan.materials.includes(plan.object));
    c.render();assert.ok(p.element('main').innerHTML.includes('评测第 '+id+' 组'));
  }
});

test('next button rotates all 30 task-2 combinations even when one gap is uniquely smallest',()=>{
  const data=structuredClone(planningFixture),objects={...task2Policy.object_phrases,brush_small:'brush',glitter_compact:'glitter'};
  data.episodes=Object.keys(objects).flatMap((o,i)=>task2Policy.destinations.flatMap((d,j)=>
    o==='lotion_bottle'&&d==='top'?[]:[record(1000+i*3+j,2,'private',{target:o,dest:d})]));
  const original=JSON.stringify(data.episodes),p=page(data),c=p.context;
  const first=c.collectionPlan(2);assert.equal(first.object+'|'+first.destination,'lotion_bottle|top');
  let last=first;
  for(let round=0;round<3;round++){
    const seen=new Set();
    for(let i=0;i<30;i++){
      const plan=round===0&&i===0?first:clickNext(p,2),key=plan.object+'|'+plan.destination;
      if(plan!==first){assert.notEqual(key,last.object+'|'+last.destination);assert.notEqual(plan.prompt,last.prompt)}
      assert.ok(!seen.has(key),'combination repeated within the rotation');seen.add(key);
      assert.equal(c.collectionPlan(2),plan,'render/copy must not advance history');
      last=plan;
    }
    assert.equal(seen.size,30);
  }
  // prep adds derived fields, so compare the source fields that affect real counts.
  assert.deepEqual(c.S.data.episodes.map(e=>[e.n,e.target,e.dest]),JSON.parse(original).map(e=>[e.n,e.target,e.dest]));
});

test('task-2 prompts use the original 48/42/43/40/43 weights and matching object/level',()=>{
  const p=page(planningFixture),c=p.context,weighted=c.weightedTemplate;
  let offset=0;
  for(const expected of task2Policy.templates){
    c.weightedTemplate=(templates)=>{
      assert.deepEqual(Array.from(templates,t=>t.weight),[48,42,43,40,43]);
      return weighted(templates,(offset+expected.weight/2)/216);
    };
    const plan=clickNext(p,2);
    assert.equal(plan.template,expected.id);
    assert.equal(plan.probability,expected.weight/216);
    assert.equal(plan.prompt,expected.text.replace('{object}',c.planningPhrases(2,task2Policy)[plan.object]).replace('{destination}',plan.destination));
    offset+=expected.weight;
  }
  const counts={};
  for(let i=0;i<216;i++){const t=weighted(task2Policy.templates,(i+.5)/216);counts[t.id]=(counts[t.id]||0)+1}
  assert.deepEqual(task2Policy.templates.map(t=>counts[t.id]),[48,42,43,40,43]);
});

test('suggestion history survives reload/data updates and remains separate for each task',()=>{
  const storage=new Map(),p=page(planningFixture,storage),c=p.context;
  const first=clickNext(p,2),t1=clickNext(p,1),t3=clickNext(p,3);
  const before=JSON.stringify(c.planningHistory(2));
  for(let i=0;i<8;i++){
    const last=c.collectionPlan(3),next=clickNext(p,3);
    assert.notEqual(next.prompt,last.prompt);assert.notEqual(next.setup,last.setup);
  }
  assert.equal(JSON.stringify(c.planningHistory(2)),before);
  assert.equal(c.collectionPlan(1),t1);
  const t1next=clickNext(p,1);assert.notEqual(t1next.prompt,t1.prompt);
  const reloaded=page(planningFixture,storage),next=reloaded.context.collectionPlan(2);
  assert.notEqual(next.object+'|'+next.destination,first.object+'|'+first.destination);
  assert.equal(reloaded.context.planningHistory(2).used.length,2);
  reloaded.context.S.data.generated='2026-11-14T18:00:00Z';
  const afterUpdate=reloaded.context.collectionPlan(2);
  assert.equal(afterUpdate,next,'a timestamp-only refresh must not consume a suggestion');
  assert.equal(reloaded.context.planningHistory(2).used.length,2);
  assert.equal(c.S.data.episodes.length,planningFixture.episodes.length);
});

test('new task data reopens old gaps and updates the visible recommendation without resetting other tasks',async()=>{
  const data=structuredClone(planningFixture),objects={...task2Policy.object_phrases,brush_small:'brush',glitter_compact:'glitter'};
  data.episodes=Object.keys(objects).flatMap((o,i)=>task2Policy.destinations.flatMap((d,j)=>
    o==='lotion_bottle'&&d==='top'?[]:[record(1000+i*3+j,2,'private',{target:o,dest:d})]));
  const p=page(data),c=p.context,first=c.collectionPlan(2);
  assert.equal(first.object+'|'+first.destination,'lotion_bottle|top');
  const second=clickNext(p,2),task1=c.collectionPlan(1),task3=c.collectionPlan(3);
  const t1History=JSON.stringify(c.planningHistory(1)),t3History=JSON.stringify(c.planningHistory(3));
  assert.equal(c.planningHistory(2).used.length,2);
  const updated=structuredClone(data);updated.generated='2026-11-14T19:00:00Z';
  updated.episodes.push(record(2000,2,'private',{target:second.object,dest:second.destination}));
  c.dataLoading=false;c.fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(updated)});
  await c.load();
  const next=c.collectionPlan(2);
  assert.equal(next.object+'|'+next.destination,'lotion_bottle|top','a previously shown but uncollected gap must become eligible after new data');
  assert.notEqual(next.prompt,second.prompt);
  assert.equal(c.planningHistory(2).used.length,1,'new task data starts a new rotation');
  assert.equal(next.basis.private_episodes,30);assert.equal(next.basis.total_episodes,30);
  assert.ok(p.element('main').innerHTML.includes('自采好数据 <b>30</b> 条'));
  assert.equal(c.collectionPlan(1),task1);assert.equal(c.collectionPlan(3),task3);
  assert.equal(JSON.stringify(c.planningHistory(1)),t1History);assert.equal(JSON.stringify(c.planningHistory(3)),t3History);
});

test('coverage changes at the same episode count and policy changes invalidate old suggestions',()=>{
  const p=page(planningFixture),c=p.context,first=c.collectionPlan(2);
  c.S.data.episodes.find(e=>e.task===2).dest='bottom';
  const next=c.collectionPlan(2);
  assert.notEqual(next.revision,first.revision);
  assert.equal(c.planningHistory(2).used.length,1);
  c.S.data.prompt_policy.tasks['2'].templates[0].weight=49;
  assert.notEqual(c.collectionPlan(2).revision,next.revision);
});

test('position grid counts independent recordings, normalizes resolutions and separates missing positions',()=>{
  const a=record(10,1,'private',{image_size:[640,480],spatial:{target_boxes:[[40,40,60,60],[45,45,65,65]],all_boxes:[[40,40,60,60],[200,200,220,220]],destination_boxes:[[400,300,420,320]]}});
  const b=record(20,1,'private',{image_size:[320,240],spatial:{target_boxes:[[20,20,30,30]],all_boxes:[[20,20,30,30]],destination_boxes:[]}});
  const missing=record(30,1,'private',{spatial:{target_boxes:[],target_status:'class_not_supported'}});
  const {context:c}=page({...fixture,episodes:[a,b,missing]});
  const stats=c.spatialStats(c.S.data.episodes,'target');
  assert.equal(stats.cells[0].records.length,2);
  assert.equal(stats.cells[0].instances,3);
  assert.equal(stats.covered,1);
  assert.equal(stats.located,2);
  assert.equal(stats.missing.length,1);
  assert.equal(c.spatialStats(c.S.data.episodes,'dest').located,1);
  assert.equal(c.spatialStats(c.S.data.episodes,'all').covered,2);
  const invalid=record(99,1,'private',{spatial:{target_boxes:[[0,0,NaN,30],[50,0,40,10],[-50,-50,-20,-20]]}});
  assert.equal(c.spatialStats([invalid],'target').located,0);
});

test('position filters and selected cells expose their contributing records',()=>{
  const spatial={status:'estimated',backgrounds:{private:{'1':'bg_private_t1.jpg'},public:{'1':'bg_public_t1.jpg'}}};
  const a=record(10,1,'private',{image_size:[320,240],spatial:{target_boxes:[[10,10,30,30]],target_status:'localized'}});
  const b=record(11,1,'public',{target:'lemon',image_size:[320,240],spatial:{target_boxes:[[100,100,120,120]],target_status:'localized'}});
  const {context:c}=page({...fixture,spatial,episodes:[a,b,record(12,2,'private')]});
  assert.equal(c.spatialRows().length,1,'defaults to private data');
  c.S.posCell=0;
  const rendered=c.vSpatial();
  assert.ok(rendered.includes('第 1 行第 1 列 · 1 条'));
  assert.ok(rendered.includes('data-ep="10"'));
  assert.ok(!rendered.includes('data-ep="11"'));
  assert.ok(rendered.includes('bg_private_t1.jpg'));
  c.S.posOrigin='all';assert.equal(c.spatialRows().length,2);
  c.S.posObject='lemon';assert.equal(c.spatialRows()[0].n,11);
  c.S.task=2;c.S.posObject='all';assert.equal(c.spatialRows()[0].n,12);
});

test('new real recordings and position maps retain counts, provenance and uncertainty',{skip:!process.env.FIELD_MERGED},()=>{
  const data=JSON.parse(fs.readFileSync(process.env.FIELD_MERGED,'utf8'));
  const own=data.episodes.filter(e=>e.origin==='private');
  assert.equal(data.episodes.length,759);assert.equal(own.length,112);
  assert.deepEqual([1,2,3].map(t=>own.filter(e=>e.task===t).length),[53,40,19]);
  assert.equal(new Set(data.episodes.map(e=>e.n)).size,759);
  assert.equal(new Set(own.map(e=>e.trajectory_sha256)).size,112);
  assert.equal(own.reduce((n,e)=>n+e.source_records.length,0),199);
  assert.equal(own.filter(e=>e.target==='brush_small').length,9);
  assert.equal(own.filter(e=>e.target==='glitter_compact').length,7);
  assert.equal(own.filter(e=>e.task<3&&!e.target).length,0);
  assert.equal(data.episodes.reduce((n,e)=>n+e.len,0),478981);
  const {context:c,element}=page(data);
  assert.ok(c.S.data.episodes.every(e=>!e.q.some(f=>f.startsWith('duplicate_of:'))),'estimated spatial proximity must not declare duplicate recordings');
  for(const task of [1,2,3])for(const origin of ['private','public','all'])for(const layer of ['target','dest','all']){
    c.S.task=task;c.S.posOrigin=origin;c.S.posLayer=layer;c.S.tab='spatial';c.S.posCell=null;c.render();
    assert.equal((element('main').innerHTML.match(/data-pos-cell=/g)||[]).length,48);
    const rows=c.spatialRows(),stats=c.spatialStats(rows,layer);
    assert.equal(stats.located+stats.missing.length,rows.length);
    assert.ok(stats.cells.every(cell=>new Set(cell.records.map(e=>e.n)).size===cell.records.length));
  }
  for(const tab of ['real','shift','coverage','analysis','quality','overview']){c.S.tab=tab;c.render();assert.ok(element('main').innerHTML.length>100)}
  c.openEp(647);assert.ok(element('dcard').innerHTML.includes('部分成功'));
});
