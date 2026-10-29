---
layout: default
title: Archive
permalink: /archive/
---

# All Diary Posts

<div class="post-archive">
  <ul style="list-style-type: none; padding: 0;">
    {% for post in site.posts %}
      <li style="margin-bottom: 20px; border-bottom: 1px solid #eaeaea; padding-bottom: 15px;">
        <span style="font-size: 0.85rem; color: #666; display: block; margin-bottom: 4px;">
          {{ post.date | date: "%B %d, %Y" }}
        </span>
        <a href="{{ post.url | relative_url }}" style="font-size: 1.1rem; font-weight: 600; text-decoration: none; color: #0366d6;">
          {{ post.title }}
        </a>
        {% if post.description %}
          <p style="margin: 5px 0 0 0; color: #444; font-size: 0.95rem;">{{ post.description }}</p>
        {% endif %}
      </li>
    {% endfor %}
  </ul>
</div>
