---
layout: default
title: Research Papers
permalink: /papers/
---

# Research Papers Collection

<ul>
  {% for paper in site.papers %}
    <li style="margin-bottom: 15px;">
      <a href="{{ paper.url | relative_url }}" style="font-weight: 600; font-size: 1.1rem;">
        {{ paper.title }}
      </a>
      <span style="display: block; color: #666; font-size: 0.85rem;">
        Published: {{ paper.date | date: "%B %d, %Y" }}
      </span>
    </li>
  {% endfor %}
</ul>
