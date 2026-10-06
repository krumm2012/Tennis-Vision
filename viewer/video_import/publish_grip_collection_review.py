"""Render the sequential fixed-9cm application and separate distance diagnostics."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image
from run_records import write,sha256


def render(output,library):
    protocol=json.loads((output/'protocol.json').read_text())
    reference=library/'85ade7a072984579831f5cb76e8e5fd3/result'
    esc=lambda value:html.escape(str(value),quote=True)
    fmt=lambda value:'—' if value is None else f'{value:.2f}'
    rows=[];cards=[];auto_rows=[];reports=[]
    for e in protocol['entries'][1:]:
        stage=output/'clips'/e['name'];dest=library/e['id']/'result'
        if not (stage/'publication.json').exists():
            rows.append(f'<tr><td>{esc(e["name"])}</td><td colspan="6">等待顺序处理</td></tr>')
            continue
        r=json.loads((stage/'application_report.json').read_text())
        a=json.loads((stage/'automatic_distance/report.json').read_text())
        viewer=e['viewer'].replace('viewer.html','grip_viewer.html')
        review=json.loads((stage/'browser_review.json').read_text()) if (stage/'browser_review.json').exists() else None
        publication=json.loads((stage/'publication.json').read_text())
        assert sha256(dest/'racket_poses.json')==r['candidate_sha256']
        assert all(sha256(dest/name)==value for name,value in publication['protected_files'].items())
        reports.append({'clip':e['name'],'viewer':viewer,'fitting':r,'automatic_distance':a,'browser_review':review})
        d=r['direction'];m=r['motion'];c=r['contact_proxy']['palm_center_gap_mm']
        state='已回放 · 待验收' if review else '已应用 · 待回放'
        rows.append(f'<tr><td><a href="{viewer}">{esc(e["name"])}</a></td><td>{state}</td><td>{r["frames"]-m["hidden"]}/{r["frames"]} · 隐藏 {m["hidden"]}</td><td>{fmt(c["median"])} / {fmt(c["p95"])} mm</td><td>{fmt(d["image_shaft_error_deg"]["median"])} / {fmt(d["image_shaft_error_deg"]["p95"])}°</td><td>{fmt(m["step_p95_deg"])}° / {fmt(m["acceleration_p95_deg"])}°</td><td>{len(r["failed_numeric_gates"])} 项待复核</td></tr>')
        ratio=a['distance_estimates_cm']['ratio'];hand=a['distance_estimates_cm']['hand_axis']
        auto_rows.append(f'<tr><td>{esc(e["name"])}</td><td>{a["racket_detection_frames"]}/{a["frames"]}</td><td>{a["visible_butt_frames"]}</td><td>{a["ratio_usable_frames"]} · {fmt(ratio["median"])} cm</td><td>{a["hand_axis_usable_frames"]} · {fmt(hand["median"])} cm</td><td>缺少独立参考 · 未验收</td></tr>')
        pictures=''
        for name,label in [('front_play.jpg','正面播放样例'),('back_grip_end.jpg','背面握拍与播放终点'),('oblique_frame168.jpg','斜视握拍特写')]:
            path=stage/name
            if path.exists():
                png='grip_collection_'+e['name'].replace('.','_')+'_'+Path(name).stem+'.png'
                with Image.open(path) as im:im.save(reference/png)
                image=f'<a href="{png}" target="_blank"><img src="{png}" loading="lazy" alt="{esc(e["name"]+label)}"></a>'
                pictures+=image if name=='front_play.jpg' else f'<details><summary>{label}</summary>{image}</details>'
        cards.append(f'<article><h2>{esc(e["name"])} · 9cm 人体＋球拍</h2><p>{state} · <a href="{viewer}">打开挥拍预览</a> · <a href="{e["viewer"]}">人体纹理对照</a></p>{pictures}</article>')
        # Give navigation links room; preserve both native and candidate rendering.
        p=dest/'grip_viewer.html';s=p.read_text()
        s=s.replace('<nav><a href="viewer.html">','<nav style="display:flex;flex-wrap:wrap;gap:14px;align-items:center"><a href="viewer.html">')
        s=s.replace('<a href="/default/viewer.html">默认视频</a>',
                    '<a href="/datasets/85ade7a072984579831f5cb76e8e5fd3/result/grip_collection_review.html">9cm 应用总览</a>')
        p.write_text(s)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>9cm 握持 · 九段挥拍进度</title><style>body{margin:0;background:#101b27;color:#e9f0f7;font:15px/1.65 system-ui}main{max-width:1400px;margin:auto;padding:24px}a{color:#84e4d1}nav{display:flex;gap:18px;flex-wrap:wrap}h1{font-size:26px}h2{font-size:21px}.notice,article,details.report{padding:18px;border:1px solid #34485e;background:#192839;border-radius:12px;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border-bottom:1px solid #34485e;text-align:left}th{color:#a9cfdd}.table{overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{display:block;max-width:100%;border-radius:8px}summary{cursor:pointer;padding:8px}small{color:#a4bbcf}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:14px}}</style><main><h1>9cm 握持 · 九段挥拍进度</h1><nav><a href="/import.html">项目视频库</a><a href="viewer.html">48.43 已验收参考</a><a href="all_video_review.html">原人体输出总览</a><a href="grip_collection_summary.json">本轮完整指标</a></nav>'''
    page+=f'<div class="notice"><strong>其余 8 段已应用 {len(reports)}/8 段。</strong><p>48.43 当前9cm版本的手—拍柄接触和拍面方向已人工验收。各段使用自己的原生手关节、相机、时间轴与球拍观测；其余段的验收状态不继承。新结果保留为候选，人体与原纹理文件未改。</p><details><summary>本轮范围与指标解释</summary><p>握点为包握区域中心在柄轴上的位置，距柄底轴心9cm；线床下缘不等同于柄端。手部用MCP/PIP走廊和13mm半径先验，未做真实皮肤碰撞验证。镜中观测未用于球拍拟合。</p><p>表中接触间隙为掌中心代理，不是皮肤表面的实际空隙。柄轴误差相对参与拟合的自动观测，是一致性检查；数值门限失败的候选需重点复核。未知帧按观测支持隐藏，未强行补齐。</p><p>共享UV纹理覆盖98.53%，仍来自离线合成；本轮没有重新生成原生人体或独立3D成片。</p></details></div>'
    from publish_practice_coaching import coaching_notice
    page+=coaching_notice()
    association=reference/'gw_training_collection_association.json'
    if association.exists():
        from gw_training_data import collection_notice
        page+=collection_notice(json.loads(association.read_text()))
    page+='<h2>固定9cm应用与播放检查</h2><div class="table"><table><thead><tr><th>视频</th><th>状态</th><th>球拍显示帧</th><th>掌中心代理 中位 / P95</th><th>图像柄轴 中位 / P95</th><th>角步长 / 加速度 P95</th><th>数值门限</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><div class="grid">'+''.join(cards)+'</div>'
    page+='<h2>自动测距 · 独立诊断</h2><p>预测未读取固定9cm设置、人工点或拟合后的球拍位姿。以下厘米值是自由估计，不应将接近9cm视为准确性证明。缺少本段人工像素或实测距离参考时，不计算准确率。</p>'
    queue=output/'automatic_distance_validation_queue.json'
    if queue.exists():
        q=json.loads(queue.read_text());write(reference/queue.name,q)
        count=sum(len(c['frames']) for c in q['clips'])
        page+=f'<p>已冻结 <a href="{queue.name}">{count} 帧独立参考核查清单</a>：每段均匀抽样并补充柄底候选、方法分歧帧，覆盖漏检。下一步核查原图握点／柄底／拍尖可见性及像素误差；厘米准确性需独立实测。清单不按接近9cm选帧。</p>'
    page+='<div class="table"><table><thead><tr><th>视频</th><th>球拍检测帧</th><th>柄底候选帧</th><th>比例法 可用帧 · 中位</th><th>手轴法 可用帧 · 中位</th><th>准确性</th></tr></thead><tbody>'+''.join(auto_rows)+'</tbody></table></div><p><small>8段数值拒绝项与完整文件身份保留在各段拟合报告。48.43 已验收版本、人工点与原历史数值报告均保持原样。</small></p></main></html>'
    (reference/'grip_collection_review.html').write_text(page)
    (output/'grip_collection_review.html').write_text(page)
    summary={'reference_acceptance':json.loads((reference/'manual_grip_face_acceptance.json').read_text()),
             'applied_clips':len(reports),'total_target_clips':8,'accepted_other_clips':0,'clips':reports}
    write(reference/'grip_collection_summary.json',summary);write(output/'summary.json',summary)
    # Keep the prior output review reachable while giving it the new preview links.
    p=reference/'all_video_review.html';s=p.read_text()
    s=s.replace('9 段均可完整播放；当前集合为人体 3D，带球拍的 48.43 单独查看。',
                '9 段人体输出均可完整播放；48.43 接触／拍面已验收，其余 8 段新增独立 9cm 球拍候选。')
    s=s.replace('本集合未应用人体、手与球拍联合拟合，3D 中没有球拍。48.43 的独立预览采用用户指定 9cm 握点；手—拍柄接触和拍面方向已由用户人工验收，其他范围另行验证。',
                '原人体纹理预览保持原样。新增 8 段球拍候选使用各段原生人体、相机与观测，固定 9cm；人体未重拟合。48.43 的手—拍柄接触和拍面方向已由用户人工验收，其余段及自动测距准确性分别验证。')
    s=s.replace('建议继续顺序：先核查人体／相机与关键帧对应，再验收 48.43 手—柄接触及拍面方向，随后扩展到其他 8 段，最后修复纹理接缝并导出成片。',
                '当前从 48.53 顺序查看新增 9cm 候选，优先复核观测稀疏和图像柄轴偏差大的帧；自动测距用独立参考单独验证。纹理接缝与成片导出仍另行处理。')
    if 'grip_collection_review.html' not in s:
        s=s.replace('<nav>','<nav><a href="grip_collection_review.html">9cm 球拍应用进度</a>',1)
    for r in reports:
        e=next(e for e in protocol['entries'] if e['name']==r['clip'])
        key=f'<nav><a href="{e["viewer"]}">'
        s=s.replace(key,f'<nav><a href="{r["viewer"]}">9cm 人体＋球拍候选</a><a href="{e["viewer"]}">') if r['viewer'] not in s else s
    p.write_text(s)
    p=reference/'viewer.html';s=p.read_text()
    if 'grip_collection_review.html' not in s:
        next_view=protocol['entries'][1]['viewer'].replace('viewer.html','grip_viewer.html')
        s=s.replace('</h1>',f'</h1><p><a href="grip_collection_review.html">其余 8 段 · 9cm 应用总览</a> · <a href="{next_view}">下一段 48.53</a></p>',1)
        p.write_text(s)
    print(json.dumps({'applied':len(reports),'browser_reviewed':sum(bool(r['browser_review']) for r in reports)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    a=p.parse_args();render(a.output,a.library)
