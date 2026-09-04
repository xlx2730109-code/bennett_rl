"""Export recorded endpoint positions as CSV and an interactive 3-D HTML plot."""

from __future__ import annotations

import csv
import json
from pathlib import Path


_COLOURS = ("#20b95b", "#1689d8", "#e5a100", "#d43b91")


def _write_html(html_path: Path, series):
    data_json = json.dumps(series, ensure_ascii=False, separators=(",", ":"))
    html_path.write_text(_HTML_TEMPLATE.replace("__TRAJECTORY_DATA__", data_json), encoding="utf-8")


def export_endpoint_trajectory(output_dir, body_names, samples):
    """Write trajectory samples to the video directory and return both output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "endpoint_trajectory_3d.csv"
    html_path = output_dir / "endpoint_trajectory_3d.html"

    series = [
        {"name": name, "colour": _COLOURS[index % len(_COLOURS)], "points": []}
        for index, name in enumerate(body_names)
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "segment", "body", "x_m", "y_m", "z_m"))
        for time_s, segment, positions in samples:
            positions = positions.tolist()
            for body_index, (x, y, z) in enumerate(positions):
                writer.writerow((f"{time_s:.6f}", segment, body_names[body_index], f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"))
                series[body_index]["points"].append(
                    [
                        round(float(x) * 1000.0, 4),
                        round(float(y) * 1000.0, 4),
                        round(float(z) * 1000.0, 4),
                        int(segment),
                        round(float(time_s), 6),
                    ]
                )

    _write_html(html_path, series)
    return csv_path, html_path


def regenerate_endpoint_trajectory_html(csv_path):
    """Rebuild the interactive HTML from an existing trajectory CSV."""
    csv_path = Path(csv_path)
    series_by_name = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            body_name = row["body"]
            if body_name not in series_by_name:
                index = len(series_by_name)
                series_by_name[body_name] = {
                    "name": body_name,
                    "colour": _COLOURS[index % len(_COLOURS)],
                    "points": [],
                }
            series_by_name[body_name]["points"].append(
                [
                    round(float(row["x_m"]) * 1000.0, 4),
                    round(float(row["y_m"]) * 1000.0, 4),
                    round(float(row["z_m"]) * 1000.0, 4),
                    int(row["segment"]),
                    round(float(row["time_s"]), 6),
                ]
            )
    if not series_by_name:
        raise ValueError(f"No trajectory rows found in: {csv_path}")
    html_path = csv_path.with_name("endpoint_trajectory_3d.html")
    _write_html(html_path, list(series_by_name.values()))
    return html_path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>足端三维轨迹</title>
<style>
  *{box-sizing:border-box} body{margin:0;background:#f5f6f8;color:#20242a;font-family:Arial,"Microsoft YaHei",sans-serif}
  .wrap{height:100vh;min-height:520px;display:flex;flex-direction:column;padding:14px 18px 18px}
  .top{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:10px}
  h1{font-size:20px;font-weight:500;margin:0 auto 0 0}.legend{display:flex;gap:12px;flex-wrap:wrap}
  label{display:flex;align-items:center;gap:5px;font-size:13px;cursor:pointer}.swatch{width:18px;height:3px;display:inline-block}
  button{border:1px solid #bcc3cc;background:#fff;color:#20242a;border-radius:5px;padding:7px 12px;cursor:pointer}
  button:hover{background:#eef1f4}.hint{font-size:12px;color:#66707c;margin-bottom:8px}
  .time-controls{display:grid;grid-template-columns:auto 90px minmax(140px,1fr) auto 90px minmax(140px,1fr);align-items:center;gap:8px 10px;margin:0 0 10px;font-size:13px}
  .time-controls input[type=number]{width:90px;border:1px solid #bcc3cc;border-radius:4px;padding:5px 7px;background:#fff;color:#20242a}
  .time-controls input[type=range]{width:100%;accent-color:#20b95b}
  .plot{position:relative;flex:1;min-height:420px;background:#fff;border:1px solid #ccd2d9}
  canvas{display:block;width:100%;height:100%;cursor:grab}canvas.dragging{cursor:grabbing}
  @media(max-width:760px){.time-controls{grid-template-columns:auto 80px 1fr}.time-controls .end-label{grid-column:1}.time-controls input[type=number]{width:80px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top"><h1>足端三维轨迹</h1><div class="legend" id="legend"></div><button id="reset">重置视角</button><button id="save">保存 PNG</button></div>
  <div class="hint">左键拖动旋转 · 右键拖动平移 · 滚轮缩放 · 坐标单位：mm</div>
  <div class="time-controls">
    <label for="start-number">起始时间（s）</label><input id="start-number" type="number"><input id="start-range" type="range">
    <label class="end-label" for="end-number">结束时间（s）</label><input id="end-number" type="number"><input id="end-range" type="range">
  </div>
  <div class="plot" id="plot"><canvas id="canvas"></canvas></div>
</div>
<script>
const series=__TRAJECTORY_DATA__;
const canvas=document.getElementById('canvas'),plot=document.getElementById('plot'),ctx=canvas.getContext('2d');
const state={yaw:-0.72,pitch:0.48,zoom:1,panX:0,panY:0,drag:false,button:0,lastX:0,lastY:0};
const all=series.flatMap(s=>s.points); const rawMins=[0,1,2].map(i=>Math.min(...all.map(p=>p[i]))); const rawMaxs=[0,1,2].map(i=>Math.max(...all.map(p=>p[i])));
const timeMin=Math.min(...all.map(p=>p[4])),timeMax=Math.max(...all.map(p=>p[4]));let selectedStart=timeMin,selectedEnd=timeMax;
const times=[...new Set(all.map(p=>p[4]))].sort((a,b)=>a-b),timeStep=times.length>1?Math.max(.001,+(times[1]-times[0]).toFixed(6)):.02;
const startNumber=document.getElementById('start-number'),endNumber=document.getElementById('end-number'),startRange=document.getElementById('start-range'),endRange=document.getElementById('end-range');
[startNumber,endNumber,startRange,endRange].forEach(el=>{el.min=timeMin;el.max=timeMax;el.step=timeStep});startNumber.value=startRange.value=timeMin;endNumber.value=endRange.value=timeMax;
function niceStep(range,target=5){const rough=Math.max(range,1e-9)/target,power=Math.pow(10,Math.floor(Math.log10(rough))),fraction=rough/power;return (fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*power}
function axisScale(lo,hi){const step=niceStep(hi-lo),min=Math.floor(lo/step)*step,max=Math.ceil(hi/step)*step,ticks=[];for(let v=min;v<=max+step*.25;v+=step)ticks.push(Math.abs(v)<step*1e-8?0:+v.toFixed(8));return {min,max,step,ticks}}
const scales=rawMins.map((v,i)=>axisScale(v,rawMaxs[i])),mins=scales.map(s=>s.min),maxs=scales.map(s=>s.max);
const centre=mins.map((v,i)=>(v+maxs[i])/2); const span=Math.max(...mins.map((v,i)=>Math.max(maxs[i]-v,1)));
series.forEach((s,i)=>{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" checked data-i="${i}"><span class="swatch" style="background:${s.colour}"></span>${s.name}`;document.getElementById('legend').appendChild(l)});
function project(p,w,h){let x=p[0]-centre[0],y=p[1]-centre[1],z=p[2]-centre[2];const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);let x1=cy*x-sy*y,y1=sy*x+cy*y;let y2=cp*y1-sp*z,z2=sp*y1+cp*z;const k=Math.min(w,h)*0.78/span*state.zoom;return [w/2+x1*k+state.panX,h/2-z2*k+state.panY,y2]}
function line3(a,b,colour,width,w,h){const p=project(a,w,h),q=project(b,w,h);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=colour;ctx.lineWidth=width;ctx.stroke()}
function polygon3(points,fill,w,h){ctx.beginPath();points.forEach((p,i)=>{const q=project(p,w,h);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.closePath();ctx.fillStyle=fill;ctx.fill()}
function label3(text,p,dx,dy,w,h,align='center'){const q=project(p,w,h);ctx.font='12px Arial';ctx.textAlign=align;ctx.textBaseline='middle';ctx.lineWidth=3;ctx.strokeStyle='rgba(255,255,255,.92)';ctx.strokeText(text,q[0]+dx,q[1]+dy);ctx.fillStyle='#4f5863';ctx.fillText(text,q[0]+dx,q[1]+dy)}
function labelOutside(text,p,distance,w,h,font='12px Arial'){const q=project(p,w,h),cx=w/2+state.panX,cy=h/2+state.panY,vx=q[0]-cx,vy=q[1]-cy,n=Math.hypot(vx,vy)||1;ctx.font=font;ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineWidth=4;ctx.strokeStyle='rgba(255,255,255,.95)';ctx.strokeText(text,q[0]+vx/n*distance,q[1]+vy/n*distance);ctx.fillStyle='#4f5863';ctx.fillText(text,q[0]+vx/n*distance,q[1]+vy/n*distance)}
function drawCoordinateBox(w,h){const [x0,y0,z0]=mins,[x1,y1,z1]=maxs;polygon3([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],'#f1f2f3',w,h);polygon3([[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],'#f6f6f7',w,h);polygon3([[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],'#f8f8f9',w,h);
 const grid='#c9cdd2';scales[0].ticks.forEach(x=>{line3([x,y0,z0],[x,y1,z0],grid,.8,w,h);line3([x,y1,z0],[x,y1,z1],grid,.8,w,h)});scales[1].ticks.forEach(y=>{line3([x0,y,z0],[x1,y,z0],grid,.8,w,h);line3([x0,y,z0],[x0,y,z1],grid,.8,w,h)});scales[2].ticks.forEach(z=>{if(z>=z1)return;line3([x0,y0,z],[x0,y1,z],grid,.8,w,h);line3([x0,y1,z],[x1,y1,z],grid,.8,w,h)});
 const edge='#7f8790',corners=[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1]],pairs=[[0,1],[1,2],[0,4]];pairs.forEach(e=>line3(corners[e[0]],corners[e[1]],edge,1,w,h));
 scales[0].ticks.forEach((v,i)=>{if(i>0)labelOutside(String(v),[v,y0,z0],15,w,h)});scales[1].ticks.forEach((v,i)=>{if(i>0)labelOutside(String(v),[x1,v,z0],15,w,h)});scales[2].ticks.forEach(v=>labelOutside(String(v),[x0,y0,v],17,w,h));
 labelOutside('X (mm)',[(x0+x1)/2,y0,z0],37,w,h,'13px Arial');labelOutside('Y (mm)',[x1,(y0+y1)/2,z0],37,w,h,'13px Arial');labelOutside('Z (mm)',[x0,y0,(z0+z1)/2],43,w,h,'13px Arial')}
function draw(){const r=plot.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(r.width*dpr));canvas.height=Math.max(1,Math.round(r.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);const w=r.width,h=r.height;ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.lineCap='round';ctx.lineJoin='round';
 drawCoordinateBox(w,h);
 document.querySelectorAll('#legend input').forEach(box=>{if(!box.checked)return;const s=series[+box.dataset.i];ctx.strokeStyle=s.colour;ctx.lineWidth=2.6;ctx.beginPath();let prev=null;for(const p of s.points){if(p[4]<selectedStart||p[4]>selectedEnd){prev=null;continue}const q=project(p,w,h);if(!prev||p[3]!==prev[3])ctx.moveTo(q[0],q[1]);else ctx.lineTo(q[0],q[1]);prev=p}ctx.stroke()});}
canvas.addEventListener('pointerdown',e=>{state.drag=true;state.button=e.button;state.lastX=e.clientX;state.lastY=e.clientY;canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});
canvas.addEventListener('pointermove',e=>{if(!state.drag)return;const dx=e.clientX-state.lastX,dy=e.clientY-state.lastY;state.lastX=e.clientX;state.lastY=e.clientY;if(state.button===2){state.panX+=dx;state.panY+=dy}else{state.yaw+=dx*.008;state.pitch=Math.max(-1.5,Math.min(1.5,state.pitch+dy*.008))}draw()});
canvas.addEventListener('pointerup',()=>{state.drag=false;canvas.classList.remove('dragging')});canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('wheel',e=>{e.preventDefault();state.zoom=Math.max(.2,Math.min(8,state.zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});
document.querySelectorAll('#legend input').forEach(x=>x.addEventListener('change',draw));document.getElementById('reset').onclick=()=>{Object.assign(state,{yaw:-.72,pitch:.48,zoom:1,panX:0,panY:0});draw()};
function setTime(which,value){value=Math.max(timeMin,Math.min(timeMax,Number(value)));if(which==='start'){selectedStart=Math.min(value,selectedEnd);startNumber.value=startRange.value=selectedStart}else{selectedEnd=Math.max(value,selectedStart);endNumber.value=endRange.value=selectedEnd}draw()}
startNumber.addEventListener('input',e=>setTime('start',e.target.value));startRange.addEventListener('input',e=>setTime('start',e.target.value));endNumber.addEventListener('input',e=>setTime('end',e.target.value));endRange.addEventListener('input',e=>setTime('end',e.target.value));
document.getElementById('save').onclick=()=>{draw();const a=document.createElement('a');a.download='endpoint_trajectory_3d.png';a.href=canvas.toDataURL('image/png');a.click()};
new ResizeObserver(draw).observe(plot);draw();
</script>
</body>
</html>'''
