"""Read-only GW training snapshots and conservative video association diagnostics.

No sync/control API is called. Missing fields remain unknown; device points and
ball_speed never become movement scores or assumed incoming machine speed.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen
from zoneinfo import ZoneInfo
from practice_scoring import number
from run_records import now, sha256, write

API_ORIGIN = 'https://51alljoin.cn:19999'
LOCAL_ZONE = ZoneInfo('Asia/Shanghai')
SERVE_TYPES = {0:'默认',1:'正手定点',2:'反手定点',3:'正手跑动',4:'反手跑动',5:'随机'}
DETAIL_FIELDS = ['serve_type','serve_speed','serve_count','serve_records','ball_speed','serve_height']


def timestamp(value):
    if not isinstance(value,str):return None
    try:
        date=datetime.fromisoformat(value.replace('Z','+00:00'))
        return date.replace(tzinfo=LOCAL_ZONE) if date.tzinfo is None else date
    except ValueError:return None


def validate_page(payload, user):
    if not isinstance(payload,dict) or payload.get('status')!='success':
        raise ValueError('GW 用户查询失败')
    data=payload.get('data')
    if not isinstance(data,dict):raise ValueError('GW 用户查询结构无效')
    info=data.get('user_info') or {}
    if str(info.get('phone_number',''))!=user:raise ValueError('GW 返回的用户与查询用户不一致')
    if not isinstance(data.get('records'),list):raise ValueError('GW 缺少训练记录列表')
    for row in data['records']:
        if not isinstance(row,dict):raise ValueError('训练记录结构无效')
        if row.get('unified_user_id') is not None and row['unified_user_id']!=info.get('unified_user_id'):
            raise ValueError('训练记录混入其他用户')
    return data


def fetch_user(user, output, opener=urlopen):
    if not re.fullmatch(r'\d{11}',user):raise ValueError('请使用完整的 11 位查询用户手机号')
    if (output/"user_snapshot.json").exists():
        raise ValueError("快照已存在，请使用新目录保留历史查询")
    output.mkdir(parents=True,exist_ok=True)
    pages=[];rows=[];seen=set();expected_total=None
    for page in range(1,101):
        url=API_ORIGIN+'/api/user/'+user+'/records?'+urlencode({'page':page,'page_size':500})
        with opener(url,timeout=30) as response:
            raw=response.read(16*1024*1024+1)
            if len(raw)>16*1024*1024:raise ValueError('GW 单页响应过大')
        payload=json.loads(raw);data=validate_page(payload,user);pagination=data.get('pagination') or {}
        if pagination.get('page')!=page:raise ValueError('GW 返回页码不一致')
        total=pagination.get('total_records')
        if type(total) is not int or total<0:raise ValueError('GW 缺少有效分页总数')
        if expected_total is None:expected_total=total
        elif total!=expected_total:raise ValueError('查询期间数据总数发生变化，请重新生成快照')
        (output/f'user_page_{page:04d}.json').write_bytes(raw)
        pages.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'page':page})
        for row in data['records']:
            key=(row.get('device_id'),row.get('game_id'))
            if None in key or key in seen:raise ValueError('分页记录缺少身份或重复，请核查 GW 数据')
            seen.add(key);rows.append(row)
        if not pagination.get('has_next'):break
    else:raise ValueError('GW 超过分页上限')
    if len(rows)!=expected_total:raise ValueError('分页数量与汇总数量不一致')
    snapshot={'schema_version':1,'queried_at':now(),'origin':API_ORIGIN,'user':user,
              'unified_user_id':data['user_info'].get('unified_user_id'),
              'pages':pages,'statistics':data.get('statistics') or {},'records':rows,
              'timezone_for_naive_timestamps':'Asia/Shanghai_assumed_not_clock_calibrated',
              'repository_reference':{'url':'https://github.com/krumm2012/hehaa-gw-data',
                                      'branch':'codex/gw-data-pro-plan',
                                      'reviewed_commit':'17c390cedaaf46ca35ccd9435ce5a4cfe7e670c4'},
              'missing_detail_fields':[k for k in DETAIL_FIELDS if not any(k in r for r in rows)]}
    write(output/'user_snapshot.json',snapshot)
    return snapshot


def machine_fields(record):
    # serve_speed is a device setting with no verified unit, not ball_speed km/h.
    interval=number(record.get('serve_interval'))
    speed=number(record.get('serve_speed'))
    direction=record.get('serve_type_text') or SERVE_TYPES.get(record.get('serve_type'))
    return {'speed':{'value':speed,'unit':'unverified_device_setting' if speed is not None else None,
                     'source_field':'serve_speed','usable_for_comparison':False},
            'frequency':{'value':interval if interval is not None and interval>0 else None,
                         'unit':'seconds_per_ball','source_field':'serve_interval'},
            'direction':{'value':direction,'unit':'training_pattern','source_field':'serve_type'},
            'spin':{'value':None,'unit':None,'source_field':None}}


def candidates(snapshot, start, end, tolerance_seconds=300):
    """Candidates require actual observed time support, never an invented game end."""
    found=[]
    for row in snapshot['records']:
        first=timestamp(row.get('created_at'))
        last=timestamp(row.get('end_time'))
        explicit_interval=first is not None and last is not None and last>=first
        serves=row.get('serve_records') or []
        observed=[timestamp(s.get('ball_serve_time') or s.get('created_at')) for s in serves if isinstance(s,dict)]
        observed=[t for t in observed if t is not None]
        nearby=[t for t in observed if start-timedelta(seconds=tolerance_seconds)<=t<=end+timedelta(seconds=tolerance_seconds)]
        if explicit_interval and first<=end and last>=start:
            reason='explicit_game_interval_overlap'
        elif nearby:
            reason='nearby_observed_serve_timestamps'
        elif first is not None and abs((first-start).total_seconds())<=tolerance_seconds:
            reason='nearby_game_start_only_no_verified_end'
        else:continue
        found.append({'device_id':row['device_id'],'game_id':row['game_id'],
                      'created_at':row.get('created_at'),'source_game_uuid':row.get('source_game_uuid'),'match_basis':reason,
                      'observed_serves_near_video':len(nearby),'machine':machine_fields(row),
                      'device_outcome_score':row.get('total_score',row.get('score')),
                      'status':'candidate_requires_clock_and_identity_confirmation'})
    return found


def capture(path):
    raw=subprocess.check_output(['ffprobe','-v','error','-show_entries','format_tags=creation_time:format=duration','-of','json',str(path)],text=True)
    meta=json.loads(raw)['format'];creation=timestamp((meta.get('tags') or {}).get('creation_time'))
    duration=number(meta.get('duration'))
    if creation is None or duration is None or duration<=0:raise ValueError('原视频缺少拍摄时间或时长')
    return {'start':creation.isoformat(),'end':(creation+timedelta(seconds=duration)).isoformat(),
            'local_start':creation.astimezone(LOCAL_ZONE).isoformat(),
            'local_end':(creation+timedelta(seconds=duration)).astimezone(LOCAL_ZONE).isoformat(),
            'basis':'original_container_creation_time_candidate_not_clock_calibrated',
            'source_sha256':sha256(path)}


def context(snapshot, video_sha, capture_time):
    start=timestamp(capture_time['start']);end=timestamp(capture_time['end'])
    records=candidates(snapshot,start,end)
    dates=[timestamp(r.get('created_at')) for r in snapshot['records']]
    dates=[d for d in dates if d is not None]
    return {'schema_version':1,'video_sha256':video_sha,'query_user_masked':'*******'+snapshot['user'][-4:],
            'source':{'origin':snapshot['origin'],'queried_at':snapshot['queried_at'],
                      'records_count':len(snapshot['records']),
                      'first_record_at':min(dates).isoformat() if dates else None,
                      'last_record_at':max(dates).isoformat() if dates else None,
                      'missing_detail_fields':snapshot['missing_detail_fields'],
                      'repository_reference':snapshot['repository_reference']},
            'capture_time':capture_time,'status':'candidates_need_confirmation' if records else 'no_matching_records',
            'candidates':records,'selected_record':None,'machine':{k:None for k in ['speed','frequency','direction','spin']},
            'comparison_ready':False,'movement_score':None,
            'limitations':['Device outcomes do not equal movement grades',
                          'Machine settings and measured ball speeds are different fields',
                          'Video/DB clocks and camera identity must be confirmed before association',
                          'Missing spin/settings remain unknown; no guessed default conditions']}


def collection_notice(association):
    clips=association.get('clips') or []
    if not clips:return ''
    start=min(c['capture']['local_start'] for c in clips)
    end=max(c['capture']['local_end'] for c in clips)
    count=sum(c['candidate_count'] for c in clips)
    message='目前没有对应记录，发球机条件保持未知。' if not count else f'存在 {count} 条时间候选，仍需确认设备身份及时间偏移。'
    return ('<div class="notice" id="gwTrainingNotice"><strong>GW 发球机训练数据已核查</strong>'
            f'<p>查询用户尾号{html.escape(association.get("user_suffix", ""))}：{association["records_count"]}条训练记录；'
            f'该用户记录截至{html.escape(association.get("last_record_at") or "未知")}。'
            f'九段视频原文件时间候选为{html.escape(start)}至{html.escape(end)}。{message}</p>'
            '<p>各段教练面板可展开“发球机训练数据关联”查看依据。设备得分不转换为动作评分；'
            '未返回或单位未验证的设置不自动填入训练条件。</p>'
            '<a href="gw_training_collection_association.json">九段匹配检查</a></div>')


def publish(snapshot, protocol, library, originals, output):
    published=[]
    entries=json.loads(protocol.read_text())['entries']
    for entry in entries:
        parts=urlsplit(entry['viewer']).path.split('/');result=library/parts[2]/'result'
        meta=json.loads((result/'mesh_meta.json').read_text())
        original=originals/entry['name']/'original.video'
        row=context(snapshot,meta['video_sha256'],capture(original));row['clip']=entry['name']
        write(output/'clips'/entry['name']/'machine_training_context.json',row)
        write(result/'machine_training_context.json',row)
        published.append({'clip':entry['name'],'dataset_id':parts[2],'status':row['status'],
                          'candidate_count':len(row['candidates']),'capture':row['capture_time'],
                          'context_sha256':sha256(result/'machine_training_context.json')})
    association={'schema_version':1,'origin':snapshot['origin'],
          'queried_at':snapshot['queried_at'],'records_count':len(snapshot['records']),
          'missing_detail_fields':snapshot['missing_detail_fields'],'clips':published,
          'selected_records':0,'practice_scores_written':0,
          'user_suffix':snapshot['user'][-4:],'last_record_at':row['source']['last_record_at']}
    write(output/'collection_association.json',association)
    parts=urlsplit(entries[0]['viewer']).path.split('/')
    reference=library/parts[2]/'result'
    write(reference/'gw_training_collection_association.json',association)
    dashboard=reference/'grip_collection_review.html'
    if dashboard.exists():
        text=re.sub(r'<div class="notice" id="gwTrainingNotice">.*?</div>', '', dashboard.read_text(), flags=re.S)
        dashboard.write_text(text.replace('<h2>固定9cm应用与播放检查</h2>',collection_notice(association)+'<h2>固定9cm应用与播放检查</h2>'))
    return published


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--user',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--originals',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,help='Reuse a frozen user snapshot without querying the remote service')
    a=p.parse_args()
    snapshot=json.loads(a.snapshot.read_text()) if a.snapshot else fetch_user(a.user,a.output/'api_snapshot')
    if snapshot.get('user')!=a.user:raise ValueError('快照用户与查询用户不一致')
    rows=publish(snapshot,a.protocol,a.library,a.originals,a.output)
    print(json.dumps({'records':len(snapshot['records']),'clips':len(rows),
                      'candidates':sum(r['candidate_count'] for r in rows),'selected':0},ensure_ascii=False))
