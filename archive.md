---
layout: default
title: Archive
permalink: /archive/
---

# Archive

<div class="post-archive">
  <h2>Research Papers</h2>
  <ul style="list-style-type: none; padding: 0;">
    {% assign sorted_papers = site.papers | sort: 'date' | reverse %}
    {% for paper in sorted_papers %}
      <li style="margin-bottom: 20px; border-bottom: 1px solid #eaeaea; padding-bottom: 15px;">
        <span style="font-size: 0.85rem; color: #666; display: block; margin-bottom: 4px;">
          {{ paper.date | date: "%B %d, %Y" }}
        </span>
        <a href="{{ paper.url | relative_url }}" style="font-size: 1.1rem; font-weight: 600; text-decoration: none; color: #0366d6;">
          {{ paper.title }}
        </a>
      </li>
    {% endfor %}
  </ul>

  {% if site.notes.size > 0 %}
    <h2 style="margin-top: 40px;">Daily Notes & Experiments</h2>
    <ul style="list-style-type: none; padding: 0;">
      {% assign sorted_notes = site.notes | sort: 'date' | reverse %}
      {% for note in sorted_notes %}
        <li style="margin-bottom: 20px; border-bottom: 1px solid #eaeaea; padding-bottom: 15px;">
          <span style="font-size: 0.85rem; color: #666; display: block; margin-bottom: 4px;">
            {{ note.date | date: "%B %d, %Y" }}
          </span>
          <a href="{{ note.url | relative_url }}" style="font-size: 1.1rem; font-weight: 600; text-decoration: none; color: #0366d6;">
            {{ note.title }}
          </a>
        </li>
      {% endfor %}
    </ul>
  {% endif %}
</div>
